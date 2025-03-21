package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"regexp"
	"strings"
	"time"
)

// 取最小值的辅助函数
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// 版本信息
const (
	AppName    = "OllamaBridge"
	AppVersion = "1.0.1" // 升级版本号以反映修复
)

// 配置结构
type Config struct {
	Port          int
	OllamaURL     string
	AuthEnabled   bool
	APIKeys       []string
	ReasoningTags []string
	Timeout       int
	LogLevel      string
	Verbose       bool
}

// OpenAI 请求模型
type OpenAIChatRequest struct {
	Model       string          `json:"model"`
	Messages    []OpenAIMessage `json:"messages"`
	Stream      bool            `json:"stream,omitempty"`
	MaxTokens   int             `json:"max_tokens,omitempty"`
	Temperature float64         `json:"temperature,omitempty"`
	User        string          `json:"user,omitempty"`
}

type OpenAIMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// Ollama 请求模型
type OllamaGenerateRequest struct {
	Model    string                 `json:"model"`
	Prompt   string                 `json:"prompt"`
	System   string                 `json:"system,omitempty"`
	Stream   bool                   `json:"stream,omitempty"`
	Raw      bool                   `json:"raw,omitempty"`
	Template string                 `json:"template,omitempty"`
	Context  []int                  `json:"context,omitempty"`
	Options  map[string]interface{} `json:"options,omitempty"`
}

type OllamaChatRequest struct {
	Model    string                 `json:"model"`
	Messages []OllamaMessage        `json:"messages"`
	Stream   bool                   `json:"stream,omitempty"`
	Options  map[string]interface{} `json:"options,omitempty"`
}

type OllamaMessage struct {
	Role    string   `json:"role"`
	Content string   `json:"content"`
	Images  []string `json:"images,omitempty"`
}

// Ollama 响应模型
type OllamaResponse struct {
	Model           string  `json:"model"`
	Response        string  `json:"response"`
	Done            bool    `json:"done"`
	Context         []int   `json:"context,omitempty"`
	TotalDuration   int64   `json:"total_duration,omitempty"`
	LoadDuration    int64   `json:"load_duration,omitempty"`
	PromptEvalCount int     `json:"prompt_eval_count,omitempty"`
	EvalCount       int     `json:"eval_count,omitempty"`
}

// OpenAI 响应模型
type OpenAIChatResponse struct {
	ID      string        `json:"id"`
	Object  string        `json:"object"`
	Created int64         `json:"created"`
	Model   string        `json:"model"`
	Choices []OpenAIChoice `json:"choices"`
	Usage   OpenAIUsage   `json:"usage"`
}

type OpenAIChoice struct {
	Index        int               `json:"index"`
	Message      OpenAIChatMessage `json:"message"`
	FinishReason string            `json:"finish_reason"`
}

type OpenAIChatMessage struct {
	Role             string `json:"role"`
	Content          string `json:"content"`
	ReasoningContent string `json:"reasoning_content,omitempty"`
}

type OpenAIUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// 流式响应模型
type ChatCompletionChunk struct {
	ID      string        `json:"id"`
	Object  string        `json:"object"`
	Created int64         `json:"created"`
	Model   string        `json:"model"`
	Choices []ChunkChoice `json:"choices"`
}

type ChunkChoice struct {
	Index        int        `json:"index"`
	Delta        ChunkDelta `json:"delta"`
	FinishReason string     `json:"finish_reason,omitempty"`
}

type ChunkDelta struct {
	Role             string `json:"role,omitempty"`
	Content          string `json:"content,omitempty"`
	ReasoningContent string `json:"reasoning_content,omitempty"`
}

// OpenAI 错误响应模型
type OpenAIError struct {
	Error struct {
		Message string `json:"message"`
		Type    string `json:"type"`
		Param   string `json:"param,omitempty"`
		Code    string `json:"code,omitempty"`
	} `json:"error"`
}

// OpenAI 模型列表响应
type OpenAIModelsList struct {
	Object string        `json:"object"`
	Data   []OpenAIModel `json:"data"`
}

type OpenAIModel struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	OwnedBy string `json:"owned_by"`
}

// 初始化配置
func initConfig() *Config {
	config := &Config{}
	
	flag.IntVar(&config.Port, "port", 8080, "服务器监听端口")
	flag.StringVar(&config.OllamaURL, "ollama-url", "http://localhost:11434", "Ollama API地址")
	flag.BoolVar(&config.AuthEnabled, "auth-enabled", false, "启用API密钥认证")
	
	apiKeys := flag.String("api-keys", "", "有效的API密钥列表，逗号分隔")
	reasoningTags := flag.String("reasoning-tags", "thinking,think,reasoning,reflection", "思考链标签，逗号分隔")
	
	flag.IntVar(&config.Timeout, "timeout", 30, "请求超时时间(秒)")
	flag.StringVar(&config.LogLevel, "log-level", "info", "日志级别(debug, info, warn, error)")
	flag.BoolVar(&config.Verbose, "verbose", false, "启用详细日志")
	
	flag.Parse()
	
	// 解析API密钥列表
	if *apiKeys != "" {
		config.APIKeys = strings.Split(*apiKeys, ",")
	}
	
	// 解析思考链标签
	if *reasoningTags != "" {
		config.ReasoningTags = strings.Split(*reasoningTags, ",")
	}
	
	return config
}

// 提取API密钥
func extractAPIKey(r *http.Request) string {
	auth := r.Header.Get("Authorization")
	if auth != "" {
		return strings.TrimPrefix(auth, "Bearer ")
	}
	
	return r.URL.Query().Get("api_key")
}

// 验证API密钥
func isValidAPIKey(key string, validKeys []string) bool {
	if key == "" || len(validKeys) == 0 {
		return false
	}
	
	for _, validKey := range validKeys {
		if key == validKey {
			return true
		}
	}
	
	return false
}

// 授权中间件
func authMiddleware(next http.Handler, config *Config) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !config.AuthEnabled || len(config.APIKeys) == 0 {
			next.ServeHTTP(w, r)
			return
		}
		
		apiKey := extractAPIKey(r)
		if !isValidAPIKey(apiKey, config.APIKeys) {
			sendError(w, "未授权: 无效的API密钥", http.StatusUnauthorized)
			return
		}
		
		next.ServeHTTP(w, r)
	})
}

// 发送OpenAI格式的错误响应
func sendError(w http.ResponseWriter, message string, status int) {
	var openAIErr OpenAIError
	openAIErr.Error.Message = message
	openAIErr.Error.Type = "server_error"
	
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(openAIErr)
}

// 提取思考链内容
func extractReasoning(content string, tags []string) string {
	var extracted strings.Builder
	
	for _, tag := range tags {
		openTag := "<" + tag + ">"
		closeTag := "</" + tag + ">"
		
		pattern := regexp.MustCompile("(?s)" + regexp.QuoteMeta(openTag) + "(.*?)" + regexp.QuoteMeta(closeTag))
		
		matches := pattern.FindAllStringSubmatch(content, -1)
		for _, match := range matches {
			if len(match) > 1 {
				extracted.WriteString(match[1])
				extracted.WriteString("\n")
			}
		}
	}
	
	return strings.TrimSpace(extracted.String())
}

// 从内容中移除思考链
func removeReasoningFromContent(content string, tags []string) string {
	result := content
	
	for _, tag := range tags {
		openTag := "<" + tag + ">"
		closeTag := "</" + tag + ">"
		
		pattern := regexp.MustCompile("(?s)" + regexp.QuoteMeta(openTag) + ".*?" + regexp.QuoteMeta(closeTag))
		
		result = pattern.ReplaceAllString(result, "")
	}
	
	// 清理多余空行和空格
	result = regexp.MustCompile(`\n{3,}`).ReplaceAllString(result, "\n\n")
	return strings.TrimSpace(result)
}

// 创建OpenAI格式的流式响应块
func createStreamChunk(id string, index int, content, reasoningContent string, done bool) *ChatCompletionChunk {
	chunk := &ChatCompletionChunk{
		ID:      id,
		Object:  "chat.completion.chunk",
		Created: time.Now().Unix(),
		Model:   "ollamabridge",
		Choices: []ChunkChoice{
			{
				Index: index,
				Delta: ChunkDelta{},
			},
		},
	}
	
	if done {
		chunk.Choices[0].FinishReason = "stop"
	} else {
		if reasoningContent != "" {
			chunk.Choices[0].Delta.ReasoningContent = reasoningContent
		}
		if content != "" {
			chunk.Choices[0].Delta.Content = content
		}
	}
	
	return chunk
}

// 创建OpenAI格式的非流式响应
func createOpenAIResponse(id, model string, content, reasoningContent string, promptTokens, completionTokens int) *OpenAIChatResponse {
	return &OpenAIChatResponse{
		ID:      id,
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   model,
		Choices: []OpenAIChoice{
			{
				Index: 0,
				Message: OpenAIChatMessage{
					Role:             "assistant",
					Content:          content,
					ReasoningContent: reasoningContent,
				},
				FinishReason: "stop",
			},
		},
		Usage: OpenAIUsage{
			PromptTokens:     promptTokens,
			CompletionTokens: completionTokens,
			TotalTokens:      promptTokens + completionTokens,
		},
	}
}

// 创建响应ID
func generateResponseID() string {
	return fmt.Sprintf("chatcmpl-%d", time.Now().UnixNano()/int64(time.Millisecond))
}

// 估算token数量 - 改进版
func estimateTokenCount(charCount int) int {
	// 更复杂的估算方法
	// 基于一般经验，英文约为1.3字符/token，中文约为1.1字符/token
	// 这里取保守估计
	return int(float64(charCount) / 3.5)
}

// 将OpenAI消息转换为Ollama提示
func openAIMessagesToOllamaPrompt(messages []OpenAIMessage) (string, string) {
	var prompt strings.Builder
	var systemPrompt string
	
	for _, msg := range messages {
		if msg.Role == "system" {
			systemPrompt = msg.Content
			continue
		}
		
		prompt.WriteString(msg.Role)
		prompt.WriteString(": ")
		prompt.WriteString(msg.Content)
		prompt.WriteString("\n")
	}
	
	return prompt.String(), systemPrompt
}

// 将OpenAI请求转换为Ollama Generate请求
func convertToOllamaGenerate(req *OpenAIChatRequest) (*OllamaGenerateRequest, error) {
	prompt, systemPrompt := openAIMessagesToOllamaPrompt(req.Messages)
	
	options := make(map[string]interface{})
	// 无论是否为默认值，总是设置温度
	options["temperature"] = req.Temperature
	
	if req.MaxTokens > 0 {
		options["num_predict"] = req.MaxTokens
	}
	
	return &OllamaGenerateRequest{
		Model:    req.Model,
		Prompt:   prompt,
		System:   systemPrompt,
		Stream:   req.Stream,
		Options:  options,
	}, nil
}

// 将OpenAI请求转换为Ollama Chat请求
func convertToOllamaChat(req *OpenAIChatRequest) (*OllamaChatRequest, error) {
	ollamaMessages := make([]OllamaMessage, 0, len(req.Messages))
	
	for _, msg := range req.Messages {
		ollamaMessages = append(ollamaMessages, OllamaMessage{
			Role:    msg.Role,
			Content: msg.Content,
		})
	}
	
	options := make(map[string]interface{})
	// 无论是否为默认值，总是设置温度
	options["temperature"] = req.Temperature
	
	if req.MaxTokens > 0 {
		options["num_predict"] = req.MaxTokens
	}
	
	return &OllamaChatRequest{
		Model:    req.Model,
		Messages: ollamaMessages,
		Stream:   req.Stream,
		Options:  options,
	}, nil
}

// 获取Ollama模型列表
func getOllamaModels(ollamaURL string) ([]string, error) {
	resp, err := http.Get(ollamaURL + "/api/tags")
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	
	var result struct {
		Models []struct {
			Name string `json:"name"`
		} `json:"models"`
	}
	
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}
	
	models := make([]string, 0, len(result.Models))
	for _, model := range result.Models {
		models = append(models, model.Name)
	}
	
	return models, nil
}

// 模型列表处理器
func modelsHandler(config *Config) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if config.Verbose {
			log.Printf("请求模型列表")
		}
		
		models, err := getOllamaModels(config.OllamaURL)
		if err != nil {
			log.Printf("获取模型列表失败: %v", err)
			sendError(w, "无法获取模型列表: "+err.Error(), http.StatusInternalServerError)
			return
		}
		
		openAIModels := make([]OpenAIModel, 0, len(models))
		now := time.Now().Unix()
		
		for _, model := range models {
			openAIModels = append(openAIModels, OpenAIModel{
				ID:      model,
				Object:  "model",
				Created: now,
				OwnedBy: "ollamabridge",
			})
		}
		
		response := OpenAIModelsList{
			Object: "list",
			Data:   openAIModels,
		}
		
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(response)
		
		if config.Verbose {
			log.Printf("返回模型列表: %d个模型", len(models))
		}
	}
}

// 处理流式响应
func handleStreamResponse(w http.ResponseWriter, resp *http.Response, config *Config, id string) {
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	
	log.Printf("开始处理流式响应...")
	
	flusher, ok := w.(http.Flusher)
	if !ok {
		sendError(w, "流式响应不支持", http.StatusInternalServerError)
		return
	}
	
	scanner := bufio.NewScanner(resp.Body)
	// 设置更大的缓冲区以处理长行
	const maxScanTokenSize = 1024 * 1024 // 1MB
	scanBuf := make([]byte, maxScanTokenSize)
	scanner.Buffer(scanBuf, maxScanTokenSize)
	
	// 用于收集部分响应
	var buffer strings.Builder
	var reasoningBuffer strings.Builder
	var prevReasoning string
	var prevContent string
	
	for scanner.Scan() {
		line := scanner.Text()
		if line == "" {
			continue
		}
		
		log.Printf("收到原始行: %s", line[:min(len(line), 100)]) // 只打印前100个字符避免日志过大
		
		// 检查是否为特殊非JSON消息
		if strings.HasPrefix(line, "data:") {
			// 可能是来自兼容端点的SSE消息
			line = strings.TrimPrefix(line, "data:")
			line = strings.TrimSpace(line)
			log.Printf("处理SSE消息，提取后: %s", line[:min(len(line), 100)])
		}
		
		// 特殊处理[DONE]消息
		if line == "[DONE]" {
			log.Printf("收到[DONE]消息，结束流式响应")
			fmt.Fprintf(w, "data: [DONE]\n\n")
			flusher.Flush()
			return
		}
		
		// 首先尝试OpenAI流式格式
		var openAIChunk struct {
			ID      string `json:"id"`
			Object  string `json:"object"`
			Created int64  `json:"created"`
			Model   string `json:"model"`
			Choices []struct {
				Index int `json:"index"`
				Delta struct {
					Role             string `json:"role"`
					Content          string `json:"content"`
					ReasoningContent string `json:"reasoning_content"`
				} `json:"delta"`
				FinishReason string `json:"finish_reason"`
			} `json:"choices"`
		}
		
		if err := json.Unmarshal([]byte(line), &openAIChunk); err == nil && 
			len(openAIChunk.Choices) > 0 {
			// 是OpenAI格式
			contentText := openAIChunk.Choices[0].Delta.Content
			reasoningText := openAIChunk.Choices[0].Delta.ReasoningContent
			isDone := openAIChunk.Choices[0].FinishReason != ""
			
			log.Printf("OpenAI格式解析成功: content=%s, reasoning_content=%s, done=%v", 
				contentText, reasoningText, isDone)
			
			// 直接发送OpenAI格式的delta（无需额外处理）
			chunk := createStreamChunk(id, 0, contentText, reasoningText, isDone)
			data, err := json.Marshal(chunk)
			if err != nil {
				log.Printf("序列化响应块失败: %v", err)
				sendError(w, "生成响应失败: "+err.Error(), http.StatusInternalServerError)
				return
			}
			
			log.Printf("发送数据块: content=%d字节, reasoning=%d字节",
				len(contentText), len(reasoningText))
			fmt.Fprintf(w, "data: %s\n\n", string(data))
			flusher.Flush()
			
			if isDone {
				log.Printf("模型指示生成完成，结束流")
				break
			}
		} else {
			// 尝试标准Ollama格式
			var responseText string
			var isDone bool
			
			var ollamaResp OllamaResponse
			if err := json.Unmarshal([]byte(line), &ollamaResp); err == nil {
				responseText = ollamaResp.Response
				isDone = ollamaResp.Done
				log.Printf("标准格式解析成功: response=%s, done=%v", 
					responseText[:min(len(responseText), 30)], isDone)
			} else {
				// 尝试解析为其他可能的格式
				var alternateFormat struct {
					Message string `json:"message"`
					Stop    bool   `json:"stop"`
				}
				if altErr := json.Unmarshal([]byte(line), &alternateFormat); altErr == nil {
					responseText = alternateFormat.Message
					isDone = alternateFormat.Stop
					log.Printf("使用备用格式解析成功: message=%s, stop=%v", 
						responseText[:min(len(responseText), 30)], isDone)
				} else {
					// 尝试作为纯文本处理
					if !strings.HasPrefix(line, "{") {
						// 可能是纯文本响应
						responseText = line
						log.Printf("将行作为纯文本处理: %s", responseText[:min(len(responseText), 50)])
					} else {
						log.Printf("无法解析JSON，但似乎是JSON格式，跳过此行")
						continue
					}
				}
			}
			
			// 如果提取到响应文本，则处理它
			if responseText != "" {
				// 检查是否包含思考链标签
				var contentForBuffer, reasoningForBuffer string
				
				// 首先尝试提取思考链
				for _, tag := range config.ReasoningTags {
					openTag := "<" + tag + ">"
					closeTag := "</" + tag + ">"
					
					// 检查是否包含思考链标签
					if strings.Contains(responseText, openTag) && strings.Contains(responseText, closeTag) {
						pattern := regexp.MustCompile("(?s)" + regexp.QuoteMeta(openTag) + "(.*?)" + regexp.QuoteMeta(closeTag))
						matches := pattern.FindAllStringSubmatch(responseText, -1)
						
						if len(matches) > 0 {
							// 含有思考链标签
							log.Printf("检测到思考链标签: %s", tag)
							
							// 提取思考内容
							for _, match := range matches {
								if len(match) > 1 {
									reasoningForBuffer += match[1] + "\n"
								}
							}
							
							// 移除思考链内容获取清洁内容
							contentForBuffer = pattern.ReplaceAllString(responseText, "")
						}
					}
				}
				
				// 如果没有检测到思考链标签，则整个内容作为普通内容
				if reasoningForBuffer == "" {
					contentForBuffer = responseText
				}
				
				// 累积响应
				if contentForBuffer != "" {
					buffer.WriteString(contentForBuffer)
				}
				if reasoningForBuffer != "" {
					reasoningBuffer.WriteString(reasoningForBuffer)
				}
				
				// 获取当前完整内容
				content := strings.TrimSpace(buffer.String())
				reasoning := strings.TrimSpace(reasoningBuffer.String())
				
				// 只发送差异部分
				var reasoningDiff, contentDiff string
				if reasoning != prevReasoning {
					if prevReasoning == "" {
						reasoningDiff = reasoning
					} else {
						reasoningDiff = strings.TrimPrefix(reasoning, prevReasoning)
					}
					prevReasoning = reasoning
				}
				
				if content != prevContent {
					if prevContent == "" {
						contentDiff = content
					} else {
						contentDiff = strings.TrimPrefix(content, prevContent)
					}
					prevContent = content
				}
				
				if contentDiff != "" || reasoningDiff != "" {
					// 构建流式响应块
					chunk := createStreamChunk(id, 0, contentDiff, reasoningDiff, isDone)
					data, err := json.Marshal(chunk)
					if err != nil {
						log.Printf("序列化响应块失败: %v", err)
						sendError(w, "生成响应失败: "+err.Error(), http.StatusInternalServerError)
						return
					}
					
					log.Printf("发送数据块: contentDiff=%d字节, reasoningDiff=%d字节",
						len(contentDiff), len(reasoningDiff))
					fmt.Fprintf(w, "data: %s\n\n", string(data))
					flusher.Flush()
				}
				
				if isDone {
					log.Printf("模型指示生成完成，结束流")
					break
				}
			}
		}
	}
	
	// 检查扫描器是否因错误而退出
	if err := scanner.Err(); err != nil {
		log.Printf("读取Ollama响应流时发生错误: %v", err)
		// 由于我们已经开始发送响应，只能记录错误，不能发送HTTP错误
	}
	
	// 发送结束标记
	log.Printf("发送结束标记[DONE]")
	fmt.Fprintf(w, "data: [DONE]\n\n")
	flusher.Flush()
}

// 处理非流式响应
func handleNonStreamResponse(w http.ResponseWriter, resp *http.Response, config *Config, id string, model string) {
	log.Printf("开始处理非流式响应...")
	
	var buffer bytes.Buffer
	_, err := io.Copy(&buffer, resp.Body)
	if err != nil {
		log.Printf("读取Ollama响应失败: %v", err)
		sendError(w, "读取Ollama响应失败: "+err.Error(), http.StatusInternalServerError)
		return
	}
	
	rawResponse := buffer.String()
	log.Printf("收到非流式响应: %s", rawResponse[:min(len(rawResponse), 200)]) // 只记录前200个字符
	
	// 直接尝试解析为OpenAI兼容格式，这是Ollama在v1/chat/completions端点返回的格式
	var openAIFormat struct {
		ID      string `json:"id"`
		Object  string `json:"object"`
		Created int64  `json:"created"` 
		Model   string `json:"model"`
		Choices []struct {
			Index   int `json:"index"`
			Message struct {
				Role    string `json:"role"`
				Content string `json:"content"`
			} `json:"message"`
			FinishReason string `json:"finish_reason"`
		} `json:"choices"`
	}
	
	var responseContent string
	var responseRole string
	
	if err := json.Unmarshal(buffer.Bytes(), &openAIFormat); err == nil && len(openAIFormat.Choices) > 0 {
		log.Printf("成功解析为OpenAI兼容格式")
		responseContent = openAIFormat.Choices[0].Message.Content
		responseRole = openAIFormat.Choices[0].Message.Role
		log.Printf("提取到内容: 角色=%s, 内容前100个字符=%s", 
			responseRole, responseContent[:min(len(responseContent), 100)])
		
		// 直接使用OpenAI格式响应
		response := &OpenAIChatResponse{
			ID:      id,
			Object:  "chat.completion",
			Created: time.Now().Unix(),
			Model:   model,
			Choices: []OpenAIChoice{
				{
					Index: 0,
					Message: OpenAIChatMessage{
						Role:    responseRole,
						Content: responseContent,
					},
					FinishReason: "stop",
				},
			},
			Usage: OpenAIUsage{
				PromptTokens:     estimateTokenCount(len(rawResponse) - len(responseContent)),
				CompletionTokens: estimateTokenCount(len(responseContent)),
				TotalTokens:      estimateTokenCount(len(rawResponse)),
			},
		}
		
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(response)
		log.Printf("成功返回OpenAI兼容格式响应")
		return
	}
	
	// 如果不是OpenAI格式，尝试标准Ollama格式
	log.Printf("不是OpenAI格式，尝试解析为Ollama标准格式")
	var ollamaResp OllamaResponse
	if err := json.Unmarshal(buffer.Bytes(), &ollamaResp); err != nil {
		log.Printf("标准格式解析失败: %v, 尝试备用格式", err)
		
		// 尝试解析为其他可能的格式
		var alternateFormat struct {
			Message string `json:"message"`
		}
		if altErr := json.Unmarshal(buffer.Bytes(), &alternateFormat); altErr == nil {
			log.Printf("使用备用格式解析成功")
			ollamaResp.Response = alternateFormat.Message
			ollamaResp.Done = true
		} else {
			// 最后尝试作为纯文本处理
			if !strings.HasPrefix(rawResponse, "{") {
				log.Printf("作为纯文本处理响应")
				ollamaResp.Response = rawResponse
				ollamaResp.Done = true
			} else {
				// 如果实在解析不了，直接尝试从raw response中提取content
				contentMatch := regexp.MustCompile(`"content":"(.*?)"`).FindStringSubmatch(rawResponse)
				if len(contentMatch) > 1 {
					log.Printf("使用正则表达式提取内容")
					ollamaResp.Response = contentMatch[1]
					ollamaResp.Done = true
				} else {
					log.Printf("所有解析方法均失败，返回原始响应")
					// 直接返回原始响应，不再尝试解析
					w.Header().Set("Content-Type", "application/json")
					w.Write(buffer.Bytes())
					return
				}
			}
		}
	}
	
	if ollamaResp.Response == "" {
		log.Printf("解析成功但响应为空，直接返回原始响应")
		// 直接返回原始响应
		w.Header().Set("Content-Type", "application/json")
		w.Write(buffer.Bytes())
		return
	}
	
	log.Printf("提取到响应内容: %s", ollamaResp.Response[:min(len(ollamaResp.Response), 100)])
	
	// 提取思考链
	reasoning := extractReasoning(ollamaResp.Response, config.ReasoningTags)
	content := removeReasoningFromContent(ollamaResp.Response, config.ReasoningTags)
	
	// 构造token数量（如果未提供则估算）
	promptTokens := ollamaResp.PromptEvalCount
	completionTokens := ollamaResp.EvalCount
	
	if promptTokens == 0 {
		// 估算提示token数量
		promptTokens = estimateTokenCount(len(ollamaResp.Response) - len(content))
	}
	
	if completionTokens == 0 {
		// 估算完成token数量
		completionTokens = estimateTokenCount(len(content))
	}
	
	// 创建OpenAI格式响应
	response := createOpenAIResponse(
		id,
		model,
		content,
		reasoning,
		promptTokens,
		completionTokens,
	)
	
	responseData, err := json.Marshal(response)
	if err != nil {
		log.Printf("序列化响应失败: %v", err)
		sendError(w, "生成响应失败: "+err.Error(), http.StatusInternalServerError)
		return
	}
	
	log.Printf("成功生成响应")
	w.Header().Set("Content-Type", "application/json")
	w.Write(responseData)
}

// 聊天完成处理器
func chatCompletionsHandler(config *Config) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		// 记录请求
		if config.Verbose {
			log.Printf("接收到聊天完成请求: %s", r.URL.Path)
		}
		
		// 只处理POST请求
		if r.Method != http.MethodPost {
			sendError(w, "方法不允许", http.StatusMethodNotAllowed)
			return
		}
		
		// 解析请求
		var openAIReq OpenAIChatRequest
		if err := json.NewDecoder(r.Body).Decode(&openAIReq); err != nil {
			log.Printf("解析请求失败: %v", err)
			sendError(w, "无效的请求JSON: "+err.Error(), http.StatusBadRequest)
			return
		}
		
		// 验证请求参数
		if openAIReq.Model == "" {
			sendError(w, "缺少必要参数: model", http.StatusBadRequest)
			return
		}
		
		if len(openAIReq.Messages) == 0 {
			sendError(w, "缺少必要参数: messages", http.StatusBadRequest)
			return
		}
		
		// 温度值检查
		if openAIReq.Temperature < 0 {
			openAIReq.Temperature = 0
		} else if openAIReq.Temperature > 2 {
			openAIReq.Temperature = 2
		}
		
		// 生成响应ID
		responseID := generateResponseID()
		
		// 是否使用兼容端点
		useCompatAPI := strings.HasPrefix(r.URL.Path, "/v1/")
		var ollamaReqBody []byte
		var ollamaEndpoint string
		var err error
		
		if useCompatAPI {
			// 使用Ollama的OpenAI兼容端点
			ollamaEndpoint = "/v1/chat/completions"
			
			// 移除非兼容参数
			cleanReq := openAIReq
			ollamaReqBody, err = json.Marshal(cleanReq)
		} else {
			// 使用Ollama原生API
			ollamaEndpoint = "/api/chat"
			ollamaReq, err := convertToOllamaChat(&openAIReq)
			if err != nil {
				log.Printf("转换请求失败: %v", err)
				sendError(w, "转换请求失败: "+err.Error(), http.StatusInternalServerError)
				return
			}
			ollamaReqBody, err = json.Marshal(ollamaReq)
		}
		
		if err != nil {
			log.Printf("构建请求失败: %v", err)
			sendError(w, "构建请求失败: "+err.Error(), http.StatusInternalServerError)
			return
		}
		
		// 创建Ollama请求
		ollamaURL := config.OllamaURL + ollamaEndpoint
		req, err := http.NewRequest(http.MethodPost, ollamaURL, bytes.NewBuffer(ollamaReqBody))
		if err != nil {
			log.Printf("创建请求失败: %v", err)
			sendError(w, "创建请求失败: "+err.Error(), http.StatusInternalServerError)
			return
		}
		
		req.Header.Set("Content-Type", "application/json")
		
		// 设置超时
		client := &http.Client{
			Timeout: time.Duration(config.Timeout) * time.Second,
		}
		
		// 发送请求到Ollama
		if config.Verbose {
			log.Printf("发送请求到Ollama: %s", ollamaURL)
		}
		
		resp, err := client.Do(req)
		if err != nil {
			log.Printf("请求Ollama失败: %v", err)
			sendError(w, "请求Ollama失败: "+err.Error(), http.StatusInternalServerError)
			return
		}
		defer resp.Body.Close()
		
		// 检查状态码
		if resp.StatusCode != http.StatusOK {
			var buffer bytes.Buffer
			io.Copy(&buffer, resp.Body)
			errMsg := fmt.Sprintf("Ollama返回错误: %s - %s", resp.Status, buffer.String())
			log.Printf(errMsg)
			sendError(w, errMsg, resp.StatusCode)
			return
		}
		
		// 处理响应
		if openAIReq.Stream {
			handleStreamResponse(w, resp, config, responseID)
		} else {
			handleNonStreamResponse(w, resp, config, responseID, openAIReq.Model)
		}
		
		if config.Verbose {
			log.Printf("成功处理请求: %s", responseID)
		}
	}
}

// 文本完成处理器（重定向到聊天完成）
func completionsHandler(config *Config) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if config.Verbose {
			log.Printf("接收到文本完成请求")
		}
		
		if r.Method != http.MethodPost {
			sendError(w, "方法不允许", http.StatusMethodNotAllowed)
			return
		}
		
		// 读取原始body
		var body map[string]interface{}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			log.Printf("解析请求失败: %v", err)
			sendError(w, "无效的请求JSON: "+err.Error(), http.StatusBadRequest)
			return
		}
		
		// 转换为chat格式
		prompt, ok := body["prompt"]
		if !ok {
			sendError(w, "缺少prompt字段", http.StatusBadRequest)
			return
		}
		
		// 确保model存在
		model, ok := body["model"].(string)
		if !ok || model == "" {
			sendError(w, "缺少或无效的model字段", http.StatusBadRequest)
			return
		}
		
		chatReq := OpenAIChatRequest{
			Model: model,
			Messages: []OpenAIMessage{
				{
					Role:    "user",
					Content: fmt.Sprintf("%v", prompt),
				},
			},
		}
		
		// 复制其他字段
		if temp, ok := body["temperature"]; ok {
			if tempFloat, ok := temp.(float64); ok {
				chatReq.Temperature = tempFloat
			}
		}
		
		if maxTokens, ok := body["max_tokens"]; ok {
			if maxTokensFloat, ok := maxTokens.(float64); ok {
				chatReq.MaxTokens = int(maxTokensFloat)
			}
		}
		
		if stream, ok := body["stream"]; ok {
			if streamBool, ok := stream.(bool); ok {
				chatReq.Stream = streamBool
			}
		}
		
		if user, ok := body["user"]; ok {
			if userStr, ok := user.(string); ok {
				chatReq.User = userStr
			}
		}
		
		// 创建新的请求
		chatBody, err := json.Marshal(chatReq)
		if err != nil {
			log.Printf("构建请求失败: %v", err)
			sendError(w, "构建请求失败: "+err.Error(), http.StatusInternalServerError)
			return
		}
		
		newReq, err := http.NewRequest(http.MethodPost, r.URL.Path, bytes.NewBuffer(chatBody))
		if err != nil {
			log.Printf("创建请求失败: %v", err)
			sendError(w, "创建请求失败: "+err.Error(), http.StatusInternalServerError)
			return
		}
		
		newReq.Header = r.Header
		
		// 调用聊天完成处理器
		handler := chatCompletionsHandler(config)
		handler(w, newReq)
	}
}

// 健康检查处理器
func healthHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{
			"status":  "ok",
			"version": AppVersion,
		})
	}
}

func main() {
	// 配置日志格式
	log.SetFlags(log.LstdFlags | log.Lmicroseconds)
	
	// 输出欢迎信息
	fmt.Printf("%s v%s - Ollama到OpenAI/DeepSeek API转换器\n", AppName, AppVersion)
	
	// 初始化配置
	config := initConfig()
	
	// 打印配置详情
	if config.Verbose {
		log.Printf("配置信息:")
		log.Printf("  端口: %d", config.Port)
		log.Printf("  Ollama地址: %s", config.OllamaURL)
		log.Printf("  认证: %v", config.AuthEnabled)
		log.Printf("  超时: %d秒", config.Timeout)
		log.Printf("  思考链标签: %v", config.ReasoningTags)
	}
	
	// 创建路由器
	mux := http.NewServeMux()
	
	// 注册路由
	mux.HandleFunc("/v1/chat/completions", chatCompletionsHandler(config))
	mux.HandleFunc("/v1/completions", completionsHandler(config))
	mux.HandleFunc("/v1/models", modelsHandler(config))
	mux.HandleFunc("/health", healthHandler())
	
	// 原生Ollama API兼容路由
	mux.HandleFunc("/api/chat", chatCompletionsHandler(config))
	mux.HandleFunc("/api/generate", completionsHandler(config))
	
	// 添加根路径处理
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/" {
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(map[string]string{
				"name":    AppName,
				"version": AppVersion,
				"status":  "running",
			})
		} else {
			http.NotFound(w, r)
		}
	})
	
	// 应用认证中间件
	var handler http.Handler = mux
	if config.AuthEnabled {
		handler = authMiddleware(mux, config)
	}
	
	// 启动服务器
	serverAddr := fmt.Sprintf(":%d", config.Port)
	fmt.Printf("启动服务器，监听地址 %s\n", serverAddr)
	fmt.Printf("接口:\n")
	fmt.Printf("  - OpenAI 兼容: http://localhost:%d/v1/chat/completions\n", config.Port)
	fmt.Printf("  - OpenAI 兼容: http://localhost:%d/v1/completions\n", config.Port)
	fmt.Printf("  - OpenAI 兼容: http://localhost:%d/v1/models\n", config.Port)
	fmt.Printf("  - 健康检查: http://localhost:%d/health\n", config.Port)
	fmt.Printf("使用Ollama API地址: %s\n", config.OllamaURL)
	
	if config.AuthEnabled {
		fmt.Printf("API密钥认证已启用\n")
	} else {
		fmt.Printf("API密钥认证已禁用\n")
	}
	
	// 创建HTTP服务器
	server := &http.Server{
		Addr:    serverAddr,
		Handler: handler,
	}
	
	// 启动服务器
	if err := server.ListenAndServe(); err != nil {
		fmt.Fprintf(os.Stderr, "启动服务器失败: %v\n", err)
		os.Exit(1)
	}
}