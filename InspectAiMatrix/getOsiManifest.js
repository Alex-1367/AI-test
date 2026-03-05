// exploreManifest.js
import fs from 'fs/promises';
import path from 'path';

// Configuration - UPDATE THESE PATHS TO MATCH YOUR SYSTEM!
const MANIFEST_PATH = '/usr/share/ollama/.ollama/models/manifests/registry.ollama.ai/library/qwen2.5-coder/1.5b';
const BLOBS_DIR = '/usr/share/ollama/.ollama/models/blobs';

async function exploreManifest() {
  console.log('🔍 Exploring Ollama Model Manifest (like Docker OCI)');
  console.log('===================================================\n');

  try {
    // 1. Read and parse the manifest file
    console.log(`📄 Reading manifest from: ${MANIFEST_PATH}`);
    const manifestContent = await fs.readFile(MANIFEST_PATH, 'utf-8');
    const manifest = JSON.parse(manifestContent);
    
    console.log('\n📋 Manifest Overview:');
    console.log(`   Schema Version: ${manifest.schemaVersion}`);
    console.log(`   Media Type: ${manifest.mediaType}`);
    console.log(`   Config Digest: ${manifest.config.digest.substring(0, 20)}...`);
    console.log(`   Config Size: ${manifest.config.size} bytes`);
    
    // 2. Display layer information
    console.log('\n📦 Model Layers (like Docker image layers):');
    console.log('   Each layer is a separate component of the AI model.\n');
    
    for (let i = 0; i < manifest.layers.length; i++) {
      const layer = manifest.layers[i];
      const digestShort = layer.digest.replace('sha256:', '').substring(0, 12);
      const blobFileName = `sha256-${layer.digest.replace('sha256:', '')}`;
      const blobPath = path.join(BLOBS_DIR, blobFileName);
      
      console.log(`   ─── Layer ${i + 1} ───────────────────────`);
      console.log(`   📌 Media Type : ${layer.mediaType}`);
      console.log(`   🆔 Digest     : sha256:${digestShort}...`);
      console.log(`   💾 Size       : ${(layer.size / 1024).toFixed(2)} KB`);
      
      // Try to get file stats to verify it exists
      try {
        const stats = await fs.stat(blobPath);
        console.log(`   ✅ File exists: ${blobFileName} (${(stats.size / 1024 / 1024).toFixed(2)} MB)`);
      } catch (err) {
        console.log(`   ❌ File missing: ${blobFileName} - Expected at ${blobPath}`);
      }
      
      // Special handling for different layer types
      if (layer.mediaType.includes('ollama.image.model')) {
        console.log(`   🧠 This is the MAIN MODEL FILE (weights & parameters)`);
      } else if (layer.mediaType.includes('ollama.image.template')) {
        console.log(`   📝 This is the PROMPT TEMPLATE`);
        // Optional: Read and display the template content
        try {
          const templateContent = await fs.readFile(blobPath, 'utf-8');
          console.log(`      First line: ${templateContent.split('\n')[0].substring(0, 80)}...`);
        } catch (e) { /* Ignore if can't read */ }
      } else if (layer.mediaType.includes('ollama.image.license')) {
        console.log(`   ⚖️  This is the LICENSE file`);
      } else if (layer.mediaType.includes('ollama.image.system')) {
        console.log(`   ⚙️  This is the SYSTEM PROMPT`);
      }
      console.log('');
    }
    
    // 3. Summary
    console.log('\n📊 Summary:');
    console.log(`   Total layers: ${manifest.layers.length}`);
    const totalSizeBytes = manifest.layers.reduce((sum, layer) => sum + layer.size, 0);
    console.log(`   Total size of all layers: ${(totalSizeBytes / 1024 / 1024 / 1024).toFixed(2)} GB`);
    console.log('\n✅ Manifest exploration complete!');
    
  } catch (error) {
    console.error('❌ Error exploring manifest:', error.message);
    console.log('\n💡 Tip: Make sure the paths in the script are correct and you have read permissions.');
  }
}

// Run the exploration
exploreManifest();