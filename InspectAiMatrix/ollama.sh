ollama pull qwen2.5-coder:1.5b
cd /usr/share/ollama/.ollama/models/manifests/registry.ollama.ai/library/
sudo ls -la
cd qwen2.5-coder
sudo ls -la
sudo cat 1.5b
sudo cat 1.5b | python3 -m json.tool 2>/dev/null || sudo cat 1.5b
olama --help
ollama -v
ollama list
echo "Hello" | ollama run qwen2.5-coder:1.5b
curl http://localhost:11434/api/generate -d '{
  "model": "qwen2.5-coder:1.5b",
  "prompt": "List ONLY JavaScript declaration keywords (var, let, const, function, class, etc.)",
  "stream": false
}' | jq -r '.response'
