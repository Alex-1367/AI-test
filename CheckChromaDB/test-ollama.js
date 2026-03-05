async function testOllama() {
  console.log('Testing Ollama with your model: qwen2.5-coder:1.5b\n');

  // Test 1: Check if model exists
  const listResponse = await fetch('http://localhost:11434/api/tags');
  const models = await listResponse.json();
  console.log('📋 Installed models:', models.models.map(m => m.name));

  // Test 2: Generate a simple response
  console.log('\n🤔 Testing generation...');
  const generateResponse = await fetch('http://localhost:11434/api/generate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: 'qwen2.5-coder:1.5b',
      prompt: 'Say "Hello, ChromaDB!" in one sentence.',
      stream: false,
    }),
  });

  const generateData = await generateResponse.json();
  console.log('✅ Generation response:', generateData.response);

  // Test 3: Test embeddings
  console.log('\n🔢 Testing embeddings...');
  const embedResponse = await fetch('http://localhost:11434/api/embeddings', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: 'qwen2.5-coder:1.5b',
      prompt: 'This is a test sentence.',
    }),
  });

  const embedData = await embedResponse.json();
  console.log('✅ Embedding generated, length:', embedData.embedding.length);
}

testOllama().catch(console.error);