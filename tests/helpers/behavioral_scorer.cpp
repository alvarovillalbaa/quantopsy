// Trusted CI-only task scorer for the pinned llama.cpp API; not application code.
#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr const char * revision = "74d4f5b041ad837153b0e90fc864b8290e01d8d5+quantopsy-target-nll-v1";
constexpr int context_tokens = 128;
constexpr int batch_tokens = 128;
constexpr int threads = 2;
constexpr std::size_t max_input_bytes = 32768;

void need(bool condition, const char * reason) {
    if (!condition) {
        throw std::runtime_error(reason);
    }
}

struct task {
    std::string id;
    std::string prompt;
    std::string target;
};

int hex_digit(char value) {
    if (value >= '0' && value <= '9') return value - '0';
    if (value >= 'a' && value <= 'f') return value - 'a' + 10;
    throw std::runtime_error("task_hex_invalid");
}

std::string unhex(const std::string & value) {
    need(value.size() <= 8192 && value.size() % 2 == 0, "task_text_bound");
    std::string result;
    for (std::size_t i = 0; i < value.size(); i += 2) {
        const char decoded = static_cast<char>(16 * hex_digit(value[i]) + hex_digit(value[i + 1]));
        need(decoded != '\0', "task_nul_refused");
        result.push_back(decoded);
    }
    return result;
}

std::vector<task> read_tasks(const char * path) {
    std::ifstream stream(path, std::ios::binary);
    need(stream.good(), "task_file_unreadable");
    std::string raw(max_input_bytes + 1, '\0');
    stream.read(&raw[0], static_cast<std::streamsize>(raw.size()));
    raw.resize(static_cast<std::size_t>(stream.gcount()));
    need(stream.eof() && !raw.empty() && raw.size() <= max_input_bytes && raw.back() == '\n', "task_file_bound_or_incomplete");
    std::istringstream lines(raw);
    std::string line;
    std::vector<task> tasks;
    while (std::getline(lines, line)) {
        const auto first = line.find('\t');
        const auto second = first == std::string::npos ? first : line.find('\t', first + 1);
        need(first != std::string::npos && second != std::string::npos && line.find('\t', second + 1) == std::string::npos,
             "task_fields_invalid");
        task current{line.substr(0, first), unhex(line.substr(first + 1, second - first - 1)), unhex(line.substr(second + 1))};
        need(!current.id.empty() && current.id.size() <= 128 && !current.target.empty(), "task_identity_or_target_invalid");
        for (const char value : current.id) {
            need((value >= 'a' && value <= 'z') || (value >= 'A' && value <= 'Z') || (value >= '0' && value <= '9')
                 || value == '.' || value == '_' || value == '-', "task_identity_invalid");
        }
        for (const auto & previous : tasks) need(previous.id != current.id, "task_identity_duplicate");
        tasks.push_back(current);
        need(tasks.size() <= 8, "task_count_bound");
    }
    need(tasks.size() == 8, "task_count_mismatch");
    return tasks;
}

std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text) {
    need(text.size() <= 8192, "tokenization_text_bound");
    // Tokenize the entire concatenation once; parse_special=false treats task
    // bytes as ordinary text. add_special=true is admitted only after the
    // model's actual BOS=true/EOS=false tokenizer behavior is checked below.
    const int required = -llama_tokenize(vocab, text.data(), static_cast<int32_t>(text.size()), nullptr, 0, true, false);
    need(required > 0 && required <= context_tokens, "context_overflow_no_truncation");
    std::vector<llama_token> result(static_cast<std::size_t>(required));
    need(llama_tokenize(vocab, text.data(), static_cast<int32_t>(text.size()), result.data(), required, true, false) == required,
         "tokenization_count_changed");
    return result;
}

struct scored_task {
    std::string id;
    std::size_t prompt_count;
    std::vector<llama_token> token_ids;
    std::vector<llama_token> target_ids;
    std::vector<llama_token> predicted_ids;
    std::vector<double> target_nll;
};

scored_task score(llama_context * ctx, const llama_vocab * vocab, const task & input) {
    const auto prefix = tokenize(vocab, input.prompt);
    auto tokens = tokenize(vocab, input.prompt + input.target);
    need(tokens.size() > prefix.size() && std::equal(prefix.begin(), prefix.end(), tokens.begin()), "prompt_target_token_boundary_mismatch");
    need(tokens.front() == llama_vocab_bos(vocab), "bos_token_not_applied");
    const int vocab_size = llama_vocab_n_tokens(vocab);
    need(vocab_size > 0 && vocab_size <= 65536, "vocabulary_bound");
    for (const auto token : tokens) need(token >= 0 && token < vocab_size, "token_id_outside_vocabulary");
    scored_task result{input.id, prefix.size(), tokens, {}, {}, {}};
    llama_kv_self_clear(ctx);
    llama_batch batch = llama_batch_init(batch_tokens, 0, 1);
    struct batch_guard {
        llama_batch * value;
        ~batch_guard() { llama_batch_free(*value); }
    } free_batch{&batch};
    need(batch.token && batch.pos && batch.n_seq_id && batch.seq_id && batch.logits, "batch_allocation_failed");
    // Position i predicts token i+1. Feed all but the final target token and
    // request only the logits that predict target tokens, including the first.
    for (std::size_t start = 0; start + 1 < tokens.size(); start += batch_tokens) {
        const std::size_t count = std::min<std::size_t>(batch_tokens, tokens.size() - start - 1);
        batch.n_tokens = static_cast<int32_t>(count);
        for (std::size_t i = 0; i < count; ++i) {
            batch.token[i] = tokens[start + i];
            batch.pos[i] = static_cast<llama_pos>(start + i);
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = start + i + 1 >= prefix.size() ? 1 : 0;
        }
        need(llama_decode(ctx, batch) == 0, "model_decode_failed");
        for (std::size_t i = 0; i < count; ++i) {
            if (batch.logits[i] == 0) continue;
            const float * logits = llama_get_logits_ith(ctx, static_cast<int32_t>(i));
            need(logits != nullptr, "target_logits_missing");
            double maximum = -std::numeric_limits<double>::infinity();
            llama_token predicted = 0;
            for (int token = 0; token < vocab_size; ++token) {
                need(std::isfinite(logits[token]), "target_logits_nonfinite");
                // First maximum wins ties, so accuracy has an explicit tie rule.
                if (static_cast<double>(logits[token]) > maximum) {
                    maximum = static_cast<double>(logits[token]);
                    predicted = token;
                }
            }
            double sum = 0;
            for (int token = 0; token < vocab_size; ++token) {
                sum += std::exp(static_cast<double>(logits[token]) - maximum);
            }
            const llama_token target = tokens[start + i + 1];
            const double nll = (maximum - static_cast<double>(logits[target])) + std::log(sum);
            need(std::isfinite(nll) && nll >= 0 && nll <= 1000, "target_nll_invalid");
            result.target_ids.push_back(target);
            result.predicted_ids.push_back(predicted);
            result.target_nll.push_back(nll);
        }
    }
    need(result.target_nll.size() == tokens.size() - prefix.size(), "target_score_coverage_mismatch");
    return result;
}

template<typename T> void numbers(std::ostream & out, const std::vector<T> & values) {
    out << '[';
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i) out << ',';
        out << values[i];
    }
    out << ']';
}

void output(const std::vector<scored_task> & tasks, int model_context, int vocab_size) {
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << std::fixed << std::setprecision(17);
    out << "{\"schema_version\":1,\"kind\":\"quantopsy.llama-target-scores\",\"engine_revision\":\"" << revision
        << "\",\"parameters\":{\"context_tokens\":128,\"batch_tokens\":128,\"threads\":2,\"seed\":0,\"add_bos\":true,\"add_eos\":false},"
        << "\"model_context_tokens\":" << model_context << ",\"vocab_tokens\":" << vocab_size << ",\"tasks\":[";
    for (std::size_t i = 0; i < tasks.size(); ++i) {
        if (i) out << ',';
        const auto & row = tasks[i];
        out << "{\"id\":\"" << row.id << "\",\"prompt_tokens\":" << row.prompt_count << ",\"token_ids\":";
        numbers(out, row.token_ids);
        out << ",\"target_ids\":";
        numbers(out, row.target_ids);
        out << ",\"predicted_ids\":";
        numbers(out, row.predicted_ids);
        out << ",\"target_nll\":[";
        for (std::size_t j = 0; j < row.target_nll.size(); ++j) {
            if (j) out << ',';
            out << '"' << row.target_nll[j] << '"';
        }
        out << "]}";
    }
    out << "]}\n";
    const auto raw = out.str();
    need(raw.size() <= 262144, "scorer_output_bound");
    std::cout << raw;
    std::cout.flush();
    need(std::cout.good(), "scorer_output_incomplete");
}
}  // namespace

int main(int argc, char ** argv) {
    try {
        need(argc == 3, "usage_scorer_model_tasks");
        const auto tasks = read_tasks(argv[2]);
        llama_backend_init();
        struct backend_guard { ~backend_guard() { llama_backend_free(); } } free_backend;
        auto model_parameters = llama_model_default_params();
        ggml_backend_dev_t no_offload_devices[] = {nullptr};
        model_parameters.devices = no_offload_devices;
        model_parameters.n_gpu_layers = 0;
        model_parameters.use_mmap = false;
        model_parameters.use_mlock = false;
        model_parameters.check_tensors = true;
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
            llama_model_load_from_file(argv[1], model_parameters), llama_model_free);
        need(model != nullptr, "model_load_failed");
        const auto * vocab = llama_model_get_vocab(model.get());
        need(vocab != nullptr && llama_vocab_get_add_bos(vocab) && !llama_vocab_get_add_eos(vocab), "tokenizer_boundary_flags_mismatch");
        need(llama_model_has_decoder(model.get()) && !llama_model_is_recurrent(model.get()), "dense_decoder_required");
        const int trained_context = llama_model_n_ctx_train(model.get());
        need(trained_context >= context_tokens, "context_exceeds_model_training_metadata");
        auto context_parameters = llama_context_default_params();
        context_parameters.n_ctx = context_tokens;
        context_parameters.n_batch = batch_tokens;
        context_parameters.n_ubatch = batch_tokens;
        context_parameters.n_seq_max = 1;
        context_parameters.n_threads = threads;
        context_parameters.n_threads_batch = threads;
        context_parameters.type_k = GGML_TYPE_F32;
        context_parameters.type_v = GGML_TYPE_F32;
        context_parameters.embeddings = false;
        context_parameters.offload_kqv = false;
        context_parameters.flash_attn = false;
        context_parameters.no_perf = true;
        std::unique_ptr<llama_context, decltype(&llama_free)> context(
            llama_init_from_model(model.get(), context_parameters), llama_free);
        need(context != nullptr && llama_n_ctx(context.get()) == context_tokens, "context_allocation_or_size_mismatch");
        std::vector<scored_task> scores;
        for (const auto & current : tasks) scores.push_back(score(context.get(), vocab, current));
        output(scores, trained_context, llama_vocab_n_tokens(vocab));
        return 0;
    } catch (const std::exception & error) {
        // Fixed error codes avoid echoing model paths, metadata or task text.
        std::cerr << "quantopsy_trusted_scorer_failed:" << error.what() << '\n';
        return 1;
    }
}
