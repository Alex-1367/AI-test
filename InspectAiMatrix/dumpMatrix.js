import fs from 'fs';

const MODEL_PATH = '/usr/share/ollama/.ollama/models/blobs/sha256-29d8c98fa6b098e200069bfb88b9508dc3e85586d20cba59f8dda9a808165104';

console.log('🔬 RAW MATRIX BYTE READER');
console.log('=' .repeat(70));

try {
    const buffer = fs.readFileSync(MODEL_PATH);
    console.log(`📊 File size: ${buffer.length} bytes`);

    // The token_embd.weight string we found at offset 5931791
    const stringOffset = 5931791;
    
    // First, let's look at the 16 bytes before and after to understand the structure
    console.log('\n📝 Bytes around token_embd.weight:');
    const start = stringOffset - 16;
    const end = stringOffset + 64;
    
    for (let i = start; i < end; i += 16) {
        const line = [];
        const hex = [];
        const ascii = [];
        
        for (let j = 0; j < 16 && i + j < buffer.length; j++) {
            const byte = buffer[i + j];
            hex.push(byte.toString(16).padStart(2, '0'));
            ascii.push(byte >= 32 && byte <= 126 ? String.fromCharCode(byte) : '.');
        }
        
        const marker = i === stringOffset ? ' <--- START OF STRING' : '';
        console.log(`${i.toString(16).padStart(8, '0')}: ${hex.join(' ').padEnd(48)} | ${ascii.join('')}${marker}`);
    }
    
    // Now let's manually parse the tensor header byte by byte
    console.log('\n📋 MANUAL TENSOR HEADER PARSING:');
    
    let pos = stringOffset;
    
    // Read the 8 bytes before the string (these should be the name length)
    const lenBytes = [];
    for (let i = 0; i < 8; i++) {
        lenBytes.push(buffer[pos + i]);
    }
    console.log(`\n8 bytes before string (name length): [${lenBytes.map(b => '0x' + b.toString(16).padStart(2, '0')).join(', ')}]`);
    
    // These bytes should be interpreted as a 64-bit little-endian integer
    // Let's calculate it manually
    let nameLen = 0;
    for (let i = 0; i < 8; i++) {
        nameLen += lenBytes[i] * Math.pow(256, i);
    }
    console.log(`Interpreted as 64-bit LE: ${nameLen}`);
    
    // The string itself
    const nameString = [];
    for (let i = 0; i < 17; i++) {
        nameString.push(String.fromCharCode(buffer[stringOffset + 8 + i]));
    }
    console.log(`String at offset ${stringOffset + 8}: "${nameString.join('')}"`);
    
    // Move past the name
    pos = stringOffset + 8 + 17;
    
    // Read next 8 bytes (number of dimensions)
    const dimCountBytes = [];
    for (let i = 0; i < 8; i++) {
        dimCountBytes.push(buffer[pos + i]);
    }
    console.log(`\nNext 8 bytes (dimension count): [${dimCountBytes.map(b => '0x' + b.toString(16).padStart(2, '0')).join(', ')}]`);
    
    let dimCount = 0;
    for (let i = 0; i < 8; i++) {
        dimCount += dimCountBytes[i] * Math.pow(256, i);
    }
    console.log(`Dimension count: ${dimCount}`);
    pos += 8;
    
    // Read dimensions (8 bytes each)
    const dims = [];
    for (let d = 0; d < dimCount; d++) {
        const dimBytes = [];
        for (let i = 0; i < 8; i++) {
            dimBytes.push(buffer[pos + i]);
        }
        let dim = 0;
        for (let i = 0; i < 8; i++) {
            dim += dimBytes[i] * Math.pow(256, i);
        }
        console.log(`Dimension ${d} bytes: [${dimBytes.map(b => '0x' + b.toString(16).padStart(2, '0')).join(', ')}] = ${dim}`);
        dims.push(dim);
        pos += 8;
    }
    
    // Read tensor type (4 bytes)
    const typeBytes = [];
    for (let i = 0; i < 4; i++) {
        typeBytes.push(buffer[pos + i]);
    }
    let tensorType = 0;
    for (let i = 0; i < 4; i++) {
        tensorType += typeBytes[i] * Math.pow(256, i);
    }
    console.log(`\nTensor type bytes: [${typeBytes.map(b => '0x' + b.toString(16).padStart(2, '0')).join(', ')}] = ${tensorType}`);
    pos += 4;
    
    console.log(`\n📐 Matrix dimensions: ${dims[0]} × ${dims[1]}`);
    console.log(`📍 Data starts at: ${pos} (0x${pos.toString(16)})`);
    
    // Now read the first 32 values of the matrix
    console.log('\n🔢 FIRST 32 VALUES (as raw bytes):');
    const firstBytes = [];
    for (let i = 0; i < 32; i++) {
        firstBytes.push(buffer[pos + i]);
    }
    console.log(`[${firstBytes.map(b => b.toString(16).padStart(2, '0')).join(' ')}]`);
    
    // Try to interpret as float16 (2 bytes per value)
    console.log('\n🔢 FIRST 16 VALUES (as float16 approximation):');
    const floatVals = [];
    for (let i = 0; i < 32 && i + 1 < 32; i += 2) {
        const low = buffer[pos + i];
        const high = buffer[pos + i + 1];
        const val = (high << 8) | low;
        // Approximate as float (simplified)
        floatVals.push(val);
    }
    console.log(`[${floatVals.join(', ')}]`);
    
    // Visualize
    console.log('\n🎨 Visual representation (first 256 bytes):');
    let visual = '';
    for (let i = 0; i < 256; i++) {
        const byte = buffer[pos + i];
        if (byte > 200) visual += '█';
        else if (byte > 150) visual += '▓';
        else if (byte > 100) visual += '▒';
        else if (byte > 50) visual += '░';
        else visual += '·';
    }
    console.log(visual);
    
    // Calculate total matrix size
    const totalValues = dims[0] * dims[1];
    const bytesPerValue = 2; // Assuming float16
    const expectedSize = totalValues * bytesPerValue;
    console.log(`\n📊 Matrix stats:`);
    console.log(`   Total values: ${totalValues.toLocaleString()}`);
    console.log(`   Expected size: ${(expectedSize / 1024 / 1024).toFixed(2)} MB`);
    console.log(`   File position: ${pos.toLocaleString()} (${(pos / buffer.length * 100).toFixed(2)}% through file)`);
    
} catch (error) {
    console.error('❌ Error:', error);
}