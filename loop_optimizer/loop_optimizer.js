/**
 * LoopOptimizer - Optimizes looping media segments for seamless playback
 * 
 * This tool analyzes audio and video clips to find optimal loop points
 * for seamless playback, helping create perfect loops for various media applications.
 */

class LoopOptimizer {
  constructor(config = {}) {
    this.config = Object.assign({
      // Default configuration
      minLoopLength: 1.0, // Minimum loop length in seconds
      maxLoopLength: 30.0, // Maximum loop length in seconds
      crossfadeDuration: 0.1, // Crossfade duration in seconds for smoother transitions
      similarityThreshold: 0.9, // Threshold for considering sections similar enough to loop (0-1)
      analysisResolution: 1024, // FFT size for audio analysis
      videoAnalysisFramerate: 5, // Frames per second to analyze for video
      outputFormat: 'wav', // Output audio format
      videoOutputFormat: 'mp4', // Output video format
      preserveOriginal: true, // Whether to keep the original file
      useWebAudioAPI: true, // Whether to use Web Audio API for processing
      useFFmpeg: false, // Whether to use FFmpeg for processing (requires backend)
    }, config);

    // Initialize Web Audio API if available and enabled
    if (this.config.useWebAudioAPI && typeof AudioContext !== 'undefined') {
      this.audioContext = new AudioContext();
    }
  }

  /**
   * Analyze audio to find optimal loop points
   * @param {ArrayBuffer|File} audioData - The audio data to analyze
   * @returns {Promise<Object>} Analysis results with potential loop points
   */
  async analyzeAudio(audioData) {
    try {
      // Decode audio data
      const audioBuffer = await this._decodeAudio(audioData);
      
      // Get audio data for analysis
      const channelData = audioBuffer.getChannelData(0); // Use first channel for analysis
      const sampleRate = audioBuffer.sampleRate;
      
      // Find potential loop points
      const loopPoints = this._findAudioLoopPoints(channelData, sampleRate);
      
      return {
        duration: audioBuffer.duration,
        sampleRate,
        numberOfChannels: audioBuffer.numberOfChannels,
        loopPoints
      };
    } catch (error) {
      console.error('Error analyzing audio:', error);
      throw error;
    }
  }

  /**
   * Create a looped version of the audio
   * @param {ArrayBuffer|File} audioData - The audio data to loop
   * @param {number} startTime - Loop start time in seconds
   * @param {number} endTime - Loop end time in seconds
   * @param {number} [duration=60] - Desired duration in seconds
   * @returns {Promise<Blob>} The looped audio as a Blob
   */
  async createAudioLoop(audioData, startTime, endTime, duration = 60) {
    try {
      // Decode audio data
      const audioBuffer = await this._decodeAudio(audioData);
      
      // Calculate samples
      const sampleRate = audioBuffer.sampleRate;
      const startSample = Math.floor(startTime * sampleRate);
      const endSample = Math.floor(endTime * sampleRate);
      const loopLengthSamples = endSample - startSample;
      
      // Calculate crossfade samples
      const crossfadeSamples = Math.floor(this.config.crossfadeDuration * sampleRate);
      
      // Calculate total output samples
      const totalOutputSamples = Math.floor(duration * sampleRate);
      
      // Create output buffer
      const outputBuffer = this.audioContext.createBuffer(
        audioBuffer.numberOfChannels,
        totalOutputSamples,
        sampleRate
      );
      
      // Process each channel
      for (let channel = 0; channel < audioBuffer.numberOfChannels; channel++) {
        const inputData = audioBuffer.getChannelData(channel);
        const outputData = outputBuffer.getChannelData(channel);
        
        // Fill the output with repeating loop segments
        let outputPosition = 0;
        
        // Add initial loop segment
        for (let i = startSample; i < endSample && outputPosition < totalOutputSamples; i++) {
          outputData[outputPosition++] = inputData[i];
        }
        
        // Add repeating loop segments with crossfade
        while (outputPosition < totalOutputSamples) {
          // Apply crossfade at loop points
          for (let i = 0; i < crossfadeSamples && i < loopLengthSamples && outputPosition < totalOutputSamples; i++) {
            const ratio = i / crossfadeSamples; // Crossfade ratio (0-1)
            const loopStartSample = startSample + i;
            
            // Crossfade between end and start of loop
            outputData[outputPosition] = inputData[loopStartSample] * ratio + 
                                         outputData[outputPosition - loopLengthSamples] * (1 - ratio);
            outputPosition++;
          }
          
          // Add remaining loop segment
          for (let i = startSample + crossfadeSamples; i < endSample && outputPosition < totalOutputSamples; i++) {
            outputData[outputPosition++] = inputData[i];
          }
        }
      }
      
      // Convert to desired format
      return this._encodeAudioBuffer(outputBuffer);
    } catch (error) {
      console.error('Error creating audio loop:', error);
      throw error;
    }
  }

  /**
   * Analyze video to find optimal loop points
   * @param {File|Blob} videoData - The video data to analyze
   * @returns {Promise<Object>} Analysis results with potential loop points
   */
  async analyzeVideo(videoData) {
    // This is a simplified version - in a real implementation,
    // this would use more sophisticated video frame analysis
    
    // Create a video element to analyze the video
    const videoElement = document.createElement('video');
    videoElement.style.display = 'none';
    document.body.appendChild(videoElement);
    
    try {
      // Set up video
      const videoUrl = URL.createObjectURL(videoData);
      videoElement.src = videoUrl;
      
      // Wait for video metadata to load
      await new Promise((resolve, reject) => {
        videoElement.onloadedmetadata = resolve;
        videoElement.onerror = reject;
      });
      
      // Get video duration
      const duration = videoElement.duration;
      
      // Calculate frame capture interval
      const captureInterval = 1 / this.config.videoAnalysisFramerate;
      
      // Mock video analysis - in a real implementation this would:
      // 1. Extract frames at regular intervals
      // 2. Compare frames using image processing
      // 3. Find segments with similar start/end frames
      // 4. Calculate similarity scores
      
      // Generate mock loop points
      const mockLoopPoints = [];
      
      // Add some potential loop points
      // In a real implementation, these would be calculated based on frame similarity
      if (duration > 3) {
        mockLoopPoints.push({
          startTime: 0,
          endTime: Math.min(10, duration),
          similarity: 0.85,
          confidence: 'medium'
        });
      }
      
      if (duration > 15) {
        mockLoopPoints.push({
          startTime: 5,
          endTime: 15,
          similarity: 0.92,
          confidence: 'high'
        });
      }
      
      // Clean up
      URL.revokeObjectURL(videoUrl);
      document.body.removeChild(videoElement);
      
      return {
        duration,
        frameRate: this._estimateFrameRate(videoElement),
        resolution: {
          width: videoElement.videoWidth,
          height: videoElement.videoHeight
        },
        loopPoints: mockLoopPoints
      };
    } catch (error) {
      // Clean up on error
      if (videoElement.parentNode) {
        document.body.removeChild(videoElement);
      }
      console.error('Error analyzing video:', error);
      throw error;
    }
  }

  /**
   * Create a looped version of the video
   * @param {File|Blob} videoData - The video data to loop
   * @param {number} startTime - Loop start time in seconds
   * @param {number} endTime - Loop end time in seconds
   * @param {number} [duration=60] - Desired duration in seconds
   * @returns {Promise<string>} URL of the looped video preview
   */
  async createVideoLoopPreview(videoData, startTime, endTime, duration = 60) {
    // This is a frontend preview method - actual video processing
    // would need to be handled by a backend service with FFmpeg
    
    try {
      // Create video preview element
      const videoElement = document.createElement('video');
      videoElement.loop = true;
      videoElement.controls = true;
      videoElement.width = 640;
      
      // Set source
      const videoUrl = URL.createObjectURL(videoData);
      videoElement.src = videoUrl;
      
      // Wait for video to load
      await new Promise((resolve, reject) => {
        videoElement.onloadedmetadata = resolve;
        videoElement.onerror = reject;
      });
      
      // Set start time based on loop points
      videoElement.currentTime = startTime;
      
      // Add event listener to reset playback position when it reaches end point
      videoElement.addEventListener('timeupdate', () => {
        if (videoElement.currentTime >= endTime) {
          videoElement.currentTime = startTime;
        }
      });
      
      return videoElement;
    } catch (error) {
      console.error('Error creating video loop preview:', error);
      throw error;
    }
  }

  /**
   * Private method to decode audio data
   * @private
   * @param {ArrayBuffer|File} audioData - Audio data to decode
   * @returns {Promise<AudioBuffer>} Decoded audio buffer
   */
  async _decodeAudio(audioData) {
    if (!this.audioContext) {
      throw new Error('Web Audio API is not supported or not enabled');
    }
    
    // If input is a File, get its ArrayBuffer
    if (audioData instanceof File || audioData instanceof Blob) {
      audioData = await audioData.arrayBuffer();
    }
    
    // Decode audio data
    return await this.audioContext.decodeAudioData(audioData);
  }

  /**
   * Private method to encode audio buffer to desired format
   * @private
   * @param {AudioBuffer} audioBuffer - Audio buffer to encode
   * @returns {Promise<Blob>} Encoded audio as Blob
   */
  async _encodeAudioBuffer(audioBuffer) {
    // This is a simplified version - in a real implementation,
    // this would use Web Audio API's MediaRecorder or a codec library
    
    // Create offline context for rendering
    const offlineContext = new OfflineAudioContext(
      audioBuffer.numberOfChannels,
      audioBuffer.length,
      audioBuffer.sampleRate
    );
    
    // Create buffer source
    const source = offlineContext.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(offlineContext.destination);
    source.start();
    
    // Render audio
    const renderedBuffer = await offlineContext.startRendering();
    
    // Convert to WAV format (simplified)
    const wavBlob = this._audioBufferToWav(renderedBuffer);
    
    return wavBlob;
  }

  /**
   * Find potential loop points in audio data
   * @private
   * @param {Float32Array} channelData - Audio channel data
   * @param {number} sampleRate - Audio sample rate
   * @returns {Array<Object>} Array of potential loop points
   */
  _findAudioLoopPoints(channelData, sampleRate) {
    // This is a simplified algorithm - a real implementation would use
    // more sophisticated analysis like FFT for frequency comparison
    
    const minSamples = this.config.minLoopLength * sampleRate;
    const maxSamples = this.config.maxLoopLength * sampleRate;
    const windowSize = Math.floor(0.05 * sampleRate); // 50ms analysis window
    
    const loopPoints = [];
    
    // Simple loop point detection based on waveform similarity
    for (let startSample = 0; startSample < channelData.length - maxSamples; startSample += windowSize) {
      for (let loopLength = minSamples; loopLength <= maxSamples; loopLength += windowSize) {
        if (startSample + loopLength >= channelData.length) {
          continue;
        }
        
        // Compare the waveform at the start and end of the potential loop
        const startSegment = channelData.slice(startSample, startSample + windowSize);
        const endSegment = channelData.slice(startSample + loopLength - windowSize, startSample + loopLength);
        
        // Calculate similarity (using RMS difference as a simple metric)
        const similarity = this._calculateSimilarity(startSegment, endSegment);
        
        if (similarity >= this.config.similarityThreshold) {
          loopPoints.push({
            startTime: startSample / sampleRate,
            endTime: (startSample + loopLength) / sampleRate,
            similarity,
            confidence: this._getSimilarityConfidence(similarity)
          });
          
          // Skip ahead to avoid similar loop points
          startSample += windowSize * 10;
          break;
        }
      }
    }
    
    // Sort by similarity (best matches first)
    return loopPoints.sort((a, b) => b.similarity - a.similarity);
  }

  /**
   * Calculate similarity between two audio segments
   * @private
   * @param {Float32Array} segment1 - First audio segment
   * @param {Float32Array} segment2 - Second audio segment
   * @returns {number} Similarity score (0-1)
   */
  _calculateSimilarity(segment1, segment2) {
    // Simple RMS difference-based similarity
    let sumSquaredDiff = 0;
    for (let i = 0; i < segment1.length; i++) {
      const diff = segment1[i] - segment2[i];
      sumSquaredDiff += diff * diff;
    }
    
    const rmsDiff = Math.sqrt(sumSquaredDiff / segment1.length);
    
    // Convert to similarity score (0-1)
    // Lower RMS difference = higher similarity
    return Math.max(0, 1 - rmsDiff * 5); // Scale factor of 5 is arbitrary
  }

  /**
   * Get confidence level from similarity score
   * @private
   * @param {number} similarity - Similarity score (0-1)
   * @returns {string} Confidence level (low, medium, high)
   */
  _getSimilarityConfidence(similarity) {
    if (similarity > 0.95) return 'high';
    if (similarity > 0.85) return 'medium';
    return 'low';
  }

  /**
   * Estimate video frame rate
   * @private
   * @param {HTMLVideoElement} videoElement - Video element
   * @returns {number} Estimated frame rate
   */
  _estimateFrameRate(videoElement) {
    // In a real implementation, this would analyze the video
    // For now, just return a common frame rate or try to get it from the video
    return videoElement.fps || 30;
  }

  /**
   * Convert AudioBuffer to WAV format
   * @private
   * @param {AudioBuffer} audioBuffer - Audio buffer to convert
   * @returns {Blob} WAV file as Blob
   */
  _audioBufferToWav(audioBuffer) {
    // Simple WAV encoder implementation
    // This is a basic implementation - a real one would be more robust
    
    const numChannels = audioBuffer.numberOfChannels;
    const sampleRate = audioBuffer.sampleRate;
    const format = 1; // PCM
    const bitDepth = 16;
    
    const bytesPerSample = bitDepth / 8;
    const blockAlign = numChannels * bytesPerSample;
    
    // Create buffer for the WAV file
    const buffer = new ArrayBuffer(44 + audioBuffer.length * blockAlign);
    const view = new DataView(buffer);
    
    // Write WAV header
    // "RIFF" chunk descriptor
    writeString(view, 0, 'RIFF');
    view.setUint32(4, 36 + audioBuffer.length * blockAlign, true);
    writeString(view, 8, 'WAVE');
    
    // "fmt " sub-chunk
    writeString(view, 12, 'fmt ');
    view.setUint32(16, 16, true); // Sub-chunk size
    view.setUint16(20, format, true); // Audio format (PCM)
    view.setUint16(22, numChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * blockAlign, true); // Byte rate
    view.setUint16(32, blockAlign, true);
    view.setUint16(34, bitDepth, true);
    
    // "data" sub-chunk
    writeString(view, 36, 'data');
    view.setUint32(40, audioBuffer.length * blockAlign, true);
    
    // Write audio data
    const offset = 44;
    const channelData = [];
    
    // Get channel data
    for (let i = 0; i < numChannels; i++) {
      channelData.push(audioBuffer.getChannelData(i));
    }
    
    // Interleave channel data and convert to 16-bit
    for (let i = 0; i < audioBuffer.length; i++) {
      for (let channel = 0; channel < numChannels; channel++) {
        const sample = Math.max(-1, Math.min(1, channelData[channel][i]));
        const sampleIndex = offset + (i * blockAlign) + (channel * bytesPerSample);
        
        // Convert float to 16-bit PCM
        const pcmSample = sample < 0 
          ? sample * 0x8000 
          : sample * 0x7FFF;
          
        view.setInt16(sampleIndex, pcmSample, true);
      }
    }
    
    // Helper function to write strings to DataView
    function writeString(view, offset, string) {
      for (let i = 0; i < string.length; i++) {
        view.setUint8(offset + i, string.charCodeAt(i));
      }
    }
    
    return new Blob([buffer], { type: 'audio/wav' });
  }
}

// Export for Node.js environments
if (typeof module !== 'undefined' && module.exports) {
  module.exports = LoopOptimizer;
}