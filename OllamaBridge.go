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

// 版本信息
const (
	AppName    = "OllamaBridge"
	AppVersion = "1.0.0"
)

// 配置结构
type Config struct {
	Port         int
	OllamaURL    string
	AuthEnabled  bool
	APIKeys      []string
	ReasoningTags []string
	Timeout      int
	LogLevel     string
	Verbose      bool
}

// OpenAI 请求模型
type OpenAIChatRequest struct {
	Model    string          `json:"model"`
	Messages []OpenAIMessage `json:"messages"`
	Stream   bool            `json:"stream,omitempty"`
	MaxTokens int            `json:"max_tokens,omitempty"`
	Temperature float64      `json:"temperature,omitempty"`
	User     string          `json:"user,omitempty"`
}

type OpenAIMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// Ollama 请求模型
type OllamaGenerateRequest struct {
	Model     string  `json:"model"`
	Prompt    string  `json:"prompt"`
	System    string  `json:"system,omitempty"`
	Stream    bool    `json:"stream,omitempty"`
	Raw       bool    `json:"raw,omitempty"`
	Template  string  `json:"template,omitempty"`
	Context   []int   `json:"context,omitempty"`
	Options   map[string]interface{} `json:"options,omitempty"`
}

type OllamaChatRequest struct {
	Model     string         `json:"model"`
	Messages  []OllamaMessage `json:"messages"`
	Stream    bool           `json:"stream,omitempty"`
	Options   map[string]interface{} `json:"options,omitempty"`
}

type OllamaMessage struct {
	Role     string   `json:"role"`
	Content  string   `json:"content"`
	Images   []string `json:"images,omitempty"`
}

// Ollama 响应模型
type OllamaResponse struct {
	Model          string  `json:"model"`
	Response       string  `json:"response"`
	Done           bool    `json:"done"`
	Context        []int   `json:"context,omitempty"`
	TotalDuration  int64   `json:"total_duration,omitempty"`
	LoadDuration   int64   `json:"load_duration,omitempty"`
	PromptEvalCount int    `json:"prompt_eval_count,omitempty"`
	EvalCount      int     `json:"eval_count,omitempty"`
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
	Index        int           `json:"index"`
	Message      OpenAIChatMessage `json:"message"`
	FinishReason string        `json:"finish_reason"`
}

type OpenAIChatMessage struct {
	Role              string `json:"role"`
	Content           string `json:"content"`
	ReasoningContent  string `json:"reasoning_content,omitempty"`
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

// OpenAI 模型列表响应
type OpenAIModelsList struct {
	Object string       `json:"object"`
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
			http.Error(w, "未授权: 无效的API密钥", http.StatusUnauthorized)
			return
		}
		
		next.ServeHTTP(w, r)
	})
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

// 估算token数量
func estimateTokenCount(charCount int) int {
	// 粗略估计，假设每4个字符约为1个token
	return charCount / 4
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
	if req.Temperature > 0 {
		options["temperature"] = req.Temperature
	}
	
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
	if req.Temperature > 0 {
		options["temperature"] = req.Temperature
	}
	
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
		models, err := getOllamaModels(config.OllamaURL)
		if err != nil {
			http.Error(w, "无法获取模型列表: "+err.Error(), http.StatusInternalServerError)
			return
		}
		
		openAIModels := make([]OpenAIModel, 0, len(models))
		now := time.Now().Unix()
		
		for _, model := range models {
			openAIModels = append(openAIModels, OpenAIModel{
				ID:      model,
				Object:  "model",
				Created: now,
				OwnedBy: "reasonbridge",
			})
		}
		
		response := OpenAIModelsList{
			Object: "list",
			Data:   openAIModels,
		}
		
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(response)
	}
}

// 处理流式响应
func handleStreamResponse(w http.ResponseWriter, resp *http.Response, config *Config, id string) {
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	
	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "流式响应不支持", http.StatusInternalServerError)
		return
	}
	
	scanner := bufio.NewScanner(resp.Body)
	
	// 用于收集部分响应
	var buffer strings.Builder
	var prevReasoning string
	var prevContent string
	
	for scanner.Scan() {
		line := scanner.Text()
		if line == "" {
			continue
		}
		
		var ollamaResp OllamaResponse
		if err := json.Unmarshal([]byte(line), &ollamaResp); err != nil {
			log.Printf("解析Ollama响应失败: %v", err)
			continue
		}
		
		// 累积响应
		buffer.WriteString(ollamaResp.Response)
		fullResponse := buffer.String()
		
		// 提取思考链
		reasoning := extractReasoning(fullResponse, config.ReasoningTags)
		content := removeReasoningFromContent(fullResponse, config.ReasoningTags)
		
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
		
		// 构建流式响应块
		chunk := createStreamChunk(id, 0, contentDiff, reasoningDiff, ollamaResp.Done)
		data, _ := json.Marshal(chunk)
		
		fmt.Fprintf(w, "data: %s\n\n", string(data))
		flusher.Flush()
		
		if ollamaResp.Done {
			break
		}
	}
	
	// 发送结束标记
	fmt.Fprintf(w, "data: [DONE]\n\n")
	flusher.Flush()
}

// 处理非流式响应
func handleNonStreamResponse(w http.ResponseWriter, resp *http.Response, config *Config, id string, model string) {
	var buffer bytes.Buffer
	_, err := io.Copy(&buffer, resp.Body)
	if err != nil {
		http.Error(w, "读取Ollama响应失败: "+err.Error(), http.StatusInternalServerError)
		return
	}
	
	var ollamaResp OllamaResponse
	if err := json.Unmarshal(buffer.Bytes(), &ollamaResp); err != nil {
		http.Error(w, "解析Ollama响应失败: "+err.Error(), http.StatusInternalServerError)
		return
	}
	
	// 提取思考链
	reasoning := extractReasoning(ollamaResp.Response, config.ReasoningTags)
	content := removeReasoningFromContent(ollamaResp.Response, config.ReasoningTags)
	
	// 创建OpenAI格式响应
	response := createOpenAIResponse(
		id,
		model,
		content,
		reasoning,
		ollamaResp.PromptEvalCount,
		ollamaResp.EvalCount,
	)
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// 聊天完成处理器
func chatCompletionsHandler(config *Config) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		// 只处理POST请求
		if r.Method != http.MethodPost {
			http.Error(w, "方法不允许", http.StatusMethodNotAllowed)
			return
		}
		
		// 解析请求
		var openAIReq OpenAIChatRequest
		if err := json.NewDecoder(r.Body).Decode(&openAIReq); err != nil {
			http.Error(w, "无效的请求JSON: "+err.Error(), http.StatusBadRequest)
			return
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
			if len(openAIReq.Messages) == 0 {
				http.Error(w, "缺少消息", http.StatusBadRequest)
				return
			}
			
			// 选择端点（generate或chat）
			useChat := true // 默认使用chat端点
			
			if useChat {
				ollamaEndpoint = "/api/chat"
				ollamaReq, err := convertToOllamaChat(&openAIReq)
				if err != nil {
					http.Error(w, "转换请求失败: "+err.Error(), http.StatusInternalServerError)
					return
				}
				ollamaReqBody, err = json.Marshal(ollamaReq)
			} else {
				ollamaEndpoint = "/api/generate"
				ollamaReq, err := convertToOllamaGenerate(&openAIReq)
				if err != nil {
					http.Error(w, "转换请求失败: "+err.Error(), http.StatusInternalServerError)
					return
				}
				ollamaReqBody, err = json.Marshal(ollamaReq)
			}
		}
		
		if err != nil {
			http.Error(w, "构建请求失败: "+err.Error(), http.StatusInternalServerError)
			return
		}
		
		// 创建Ollama请求
		ollamaURL := config.OllamaURL + ollamaEndpoint
		req, err := http.NewRequest(http.MethodPost, ollamaURL, bytes.NewBuffer(ollamaReqBody))
		if err != nil {
			http.Error(w, "创建请求失败: "+err.Error(), http.StatusInternalServerError)
			return
		}
		
		req.Header.Set("Content-Type", "application/json")
		
		// 设置超时
		client := &http.Client{
			Timeout: time.Duration(config.Timeout) * time.Second,
		}
		
		// 发送请求到Ollama
		resp, err := client.Do(req)
		if err != nil {
			http.Error(w, "请求Ollama失败: "+err.Error(), http.StatusInternalServerError)
			return
		}
		defer resp.Body.Close()
		
		// 检查状态码
		if resp.StatusCode != http.StatusOK {
			var buffer bytes.Buffer
			io.Copy(&buffer, resp.Body)
			http.Error(w, fmt.Sprintf("Ollama返回错误: %s - %s", resp.Status, buffer.String()), resp.StatusCode)
			return
		}
		
		// 处理响应
		if openAIReq.Stream {
			handleStreamResponse(w, resp, config, responseID)
		} else {
			handleNonStreamResponse(w, resp, config, responseID, openAIReq.Model)
		}
	}
}

// 文本完成处理器（重定向到聊天完成）
func completionsHandler(config *Config) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "方法不允许", http.StatusMethodNotAllowed)
			return
		}
		
		// 读取原始body
		var body map[string]interface{}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			http.Error(w, "无效的请求JSON: "+err.Error(), http.StatusBadRequest)
			return
		}
		
		// 转换为chat格式
		prompt, ok := body["prompt"]
		if !ok {
			http.Error(w, "缺少prompt字段", http.StatusBadRequest)
			return
		}
		
		chatReq := OpenAIChatRequest{
			Model: body["model"].(string),
			Messages: []OpenAIMessage{
				{
					Role:    "user",
					Content: fmt.Sprintf("%v", prompt),
				},
			},
		}
		
		// 复制其他字段
		if temp, ok := body["temperature"]; ok {
			chatReq.Temperature, _ = temp.(float64)
		}
		
		if maxTokens, ok := body["max_tokens"]; ok {
			chatReq.MaxTokens, _ = maxTokens.(int)
		}
		
		if stream, ok := body["stream"]; ok {
			chatReq.Stream, _ = stream.(bool)
		}
		
		if user, ok := body["user"]; ok {
			chatReq.User, _ = user.(string)
		}
		
		// 创建新的请求
		chatBody, _ := json.Marshal(chatReq)
		newReq, _ := http.NewRequest(http.MethodPost, r.URL.Path, bytes.NewBuffer(chatBody))
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
			"status": "ok",
			"version": AppVersion,
		})
	}
}

func main() {
	// 输出欢迎信息
	fmt.Printf("%s v%s - Ollama到OpenAI/DeepSeek API转换器\n", AppName, AppVersion)
	
	// 初始化配置
	config := initConfig()
	
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
