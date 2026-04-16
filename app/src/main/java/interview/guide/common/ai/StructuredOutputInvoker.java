package interview.guide.common.ai;

import interview.guide.common.exception.BusinessException;
import interview.guide.common.exception.ErrorCode;
import org.slf4j.Logger;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.converter.BeanOutputConverter;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

import java.util.regex.Pattern;

/**
 * 统一封装结构化输出调用与重试策略。
 *
 * 设计要点：
 * 1. 首次调用就附加严格的 JSON 格式约束，而非等重试时才加
 * 2. 对 AI 原始输出进行预处理，清理可能导致解析失败的控制字符
 * 3. 解析失败时尝试修复常见格式问题后重新解析
 */
@Component
public class StructuredOutputInvoker {

    /**
     * 严格的 JSON 格式指令 - 首次调用就附加到系统提示词
     */
    private static final String STRICT_JSON_INSTRUCTION = """

[JSON格式强制要求]
1. 只返回一个 JSON 对象，禁止任何 Markdown 代码块标记
2. 所有字符串值内部的英文双引号必须转义正确
3. 禁止在字符串中出现未转义的反斜杠（除非作为转义符）
4. 禁止在字符串中出现控制字符（换行符、制表符等），请用空格或文字描述替代
5. 如需引用或强调内容，请使用中文书名号《》或单引号『』
6. 确保 JSON 结构完整：所有 { 与 } 配对，所有 [ 与 ] 配对
7. 输出前自检：你的输出能否被标准 JSON 解析器正确解析？
""";

    /**
     * 控制字符模式：匹配 ASCII 控制字符（0x00-0x1F，但允许空格、制表符、换行）
     */
    private static final Pattern CONTROL_CHARS = Pattern.compile("[\\x00-\\x08\\x0B\\x0C\\x0E-\\x1F]");

    private final int maxAttempts;
    private final boolean includeLastErrorInRetryPrompt;

    public StructuredOutputInvoker(
        @Value("${app.ai.structured-max-attempts:2}") int maxAttempts,
        @Value("${app.ai.structured-include-last-error:true}") boolean includeLastErrorInRetryPrompt
    ) {
        this.maxAttempts = Math.max(1, maxAttempts);
        this.includeLastErrorInRetryPrompt = includeLastErrorInRetryPrompt;
    }

    public <T> T invoke(
        ChatClient chatClient,
        String systemPromptWithFormat,
        String userPrompt,
        BeanOutputConverter<T> outputConverter,
        ErrorCode errorCode,
        String errorPrefix,
        String logContext,
        Logger log
    ) {
        // 首次调用就附加严格的 JSON 格式约束
        String enhancedSystemPrompt = systemPromptWithFormat + STRICT_JSON_INSTRUCTION;

        Exception lastError = null;
        String lastRawResponse = null;

        for (int attempt = 1; attempt <= maxAttempts; attempt++) {
            String attemptSystemPrompt = (attempt == 1)
                ? enhancedSystemPrompt
                : buildRetrySystemPrompt(enhancedSystemPrompt, lastError, lastRawResponse);

            try {
                // 获取 AI 原始响应
                String rawResponse = chatClient.prompt()
                    .system(attemptSystemPrompt)
                    .user(userPrompt)
                    .call()
                    .content();

                lastRawResponse = rawResponse;

                // 预处理：清理可能导致解析失败的内容
                String cleanedResponse = preprocessJsonResponse(rawResponse, log);

                // 尝试解析
                return outputConverter.convert(cleanedResponse);

            } catch (Exception e) {
                lastError = e;
                log.warn("{}结构化解析失败，准备重试: attempt={}, error={}", logContext, attempt, e.getMessage());

                // 尝试修复后重新解析（仅限非首次尝试或明显的可修复问题）
                if (lastRawResponse != null && attempt == maxAttempts) {
                    try {
                        String repaired = attemptJsonRepair(lastRawResponse, log);
                        if (repaired != null) {
                            log.info("{}尝试修复 JSON 后重新解析", logContext);
                            return outputConverter.convert(repaired);
                        }
                    } catch (Exception repairError) {
                        log.warn("{}JSON 修复后仍解析失败: {}", logContext, repairError.getMessage());
                    }
                }
            }
        }

        throw new BusinessException(
            errorCode,
            errorPrefix + (lastError != null ? lastError.getMessage() : "unknown")
        );
    }

    /**
     * 预处理 AI 原始响应，清理可能导致 JSON 解析失败的内容
     */
    private String preprocessJsonResponse(String rawResponse, Logger log) {
        if (rawResponse == null || rawResponse.isEmpty()) {
            return rawResponse;
        }

        String result = rawResponse;

        // 1. 移除 Markdown 代码块标记
        result = result.replaceAll("^```json\\s*", "");
        result = result.replaceAll("^```\\s*", "");
        result = result.replaceAll("\\s*```$", "");

        // 2. 清理控制字符（保留换行和制表符）
        String before = result;
        result = CONTROL_CHARS.matcher(result).replaceAll("");
        if (!result.equals(before)) {
            log.debug("已清理响应中的控制字符");
        }

        // 3. 移除 BOM 和不可见字符
        result = result.replace("\uFEFF", "");
        result = result.trim();

        return result;
    }

    /**
     * 尝试修复常见的 JSON 格式问题
     */
    private String attemptJsonRepair(String rawResponse, Logger log) {
        if (rawResponse == null) {
            return null;
        }

        String cleaned = preprocessJsonResponse(rawResponse, log);

        // 检查是否被截断（末尾不是 } 或 ]）
        String trimmed = cleaned.trim();
        if (!trimmed.endsWith("}") && !trimmed.endsWith("]")) {
            log.warn("JSON 响应可能被截断，末尾: {}",
                trimmed.length() > 50 ? trimmed.substring(trimmed.length() - 50) : trimmed);
            return null;  // 截断的 JSON 无法简单修复
        }

        // 检查括号是否匹配
        int braceCount = 0;
        int bracketCount = 0;
        boolean inString = false;
        boolean escape = false;

        for (char c : trimmed.toCharArray()) {
            if (escape) {
                escape = false;
                continue;
            }
            if (c == '\\') {
                escape = true;
                continue;
            }
            if (c == '"') {
                inString = !inString;
                continue;
            }
            if (!inString) {
                if (c == '{') braceCount++;
                else if (c == '}') braceCount--;
                else if (c == '[') bracketCount++;
                else if (c == ']') bracketCount--;
            }
        }

        if (braceCount != 0 || bracketCount != 0) {
            log.warn("JSON 括号不匹配: 大括号={}, 中括号={}", braceCount, bracketCount);
            return null;
        }

        // 如果结构看起来正确，返回清理后的版本
        return cleaned;
    }

    private String buildRetrySystemPrompt(String systemPromptWithFormat, Exception lastError, String lastRawResponse) {
        StringBuilder prompt = new StringBuilder(systemPromptWithFormat)
            .append("\n\n[重要] 上次输出解析失败，请严格遵守以下规则重新输出：\n")
            .append("1. 绝对禁止在字符串中使用未转义的双引号\n")
            .append("2. 禁止使用 Markdown 代码块\n")
            .append("3. 确保输出是合法的 JSON 格式");

        if (includeLastErrorInRetryPrompt && lastError != null && lastError.getMessage() != null) {
            prompt.append("\n失败原因：")
                .append(sanitizeErrorMessage(lastError.getMessage()));
        }

        // 如果有上次的原始响应，提供截断参考（帮助 AI 理解问题）
        if (lastRawResponse != null && lastRawResponse.length() > 20) {
            String snippet = lastRawResponse.length() > 100
                ? lastRawResponse.substring(0, 50) + "..." + lastRawResponse.substring(lastRawResponse.length() - 50)
                : lastRawResponse;
            prompt.append("\n上次输出片段：").append(snippet);
        }

        return prompt.toString();
    }

    private String sanitizeErrorMessage(String message) {
        String oneLine = message.replace('\n', ' ').replace('\r', ' ').trim();
        if (oneLine.length() > 200) {
            return oneLine.substring(0, 200) + "...";
        }
        return oneLine;
    }
}
