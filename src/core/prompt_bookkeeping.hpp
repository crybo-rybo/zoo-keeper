/**
 * @file prompt_bookkeeping.hpp
 * @brief Pure helpers for incremental prompt checkpoint invalidation rules.
 */

#pragma once

namespace zoo::core {

enum class PromptHistoryMutation {
    Append,
    Rewrite,
    Reset,
};

[[nodiscard]] inline bool
history_mutation_requires_kv_reset(PromptHistoryMutation mutation) noexcept {
    return mutation != PromptHistoryMutation::Append;
}

inline void note_history_mutation(PromptHistoryMutation mutation, bool& cached_messages_dirty,
                                  int& committed_prompt_len) noexcept {
    cached_messages_dirty = true;
    if (history_mutation_requires_kv_reset(mutation)) {
        committed_prompt_len = 0;
    }
}

[[nodiscard]] inline bool rendered_prompt_requires_kv_reset(int committed_prompt_len,
                                                            int rendered_prompt_len) noexcept {
    return rendered_prompt_len < committed_prompt_len;
}

inline void commit_rendered_prompt(int& committed_prompt_len, int rendered_prompt_len) noexcept {
    if (rendered_prompt_len > 0) {
        committed_prompt_len = rendered_prompt_len;
    }
}

} // namespace zoo::core
