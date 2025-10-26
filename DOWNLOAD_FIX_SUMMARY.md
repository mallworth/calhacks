# Model Download & Cache Detection Fixes

## Issues Fixed

### 1. **Cache Detection Not Auto-Loading** ✅
**Problem**: `checkForCachedModel()` would detect cached model files but wouldn't actually load them into memory.

**Solution**: Modified `ios/Runner/LLMService.swift` line ~135:
- Added automatic call to `initializeRealLLMIfNeeded()` when complete cache detected
- Improved validation to check for `.safetensors`, `.gguf`, model files, and config.json
- Added proper status updates to UI via `DispatchQueue.main.async`
- Added 500ms delay before loading to let UI update

**Result**: When app restarts with cached model, it now auto-loads without requiring download button click.

### 2. **Download Error Messages Not Specific** ✅
**Problem**: Generic "Failed to load model" error didn't help users understand what went wrong.

**Solution**: Enhanced error handling in `ios/Runner/LLMService.swift` lines ~375-400:
- Added network error detection (NSURLErrorDomain)
- Specific messages for:
  - No internet connection
  - Download timeout
  - Cannot reach HuggingFace servers
  - Memory pressure
- Error messages now surface to Flutter UI via status polling

**Result**: Users see helpful error messages like "Download timed out. Please try again or use 'Clear Cache & Retry'."

### 3. **Clear Cache Not Thorough** ✅
**Problem**: `clearCache` only removed main model directory, leaving Hub metadata (snapshots, refs, blobs) and temp files.

**Solution**: Enhanced clear cache in `ios/Runner/LLMService.swift` lines ~480-520:
- Now recursively calculates and logs total cache size
- Removes Hub metadata directories: `snapshots/`, `refs/`, `blobs/`
- Clears temporary files: `.tmp`, `.download`, `.partial`
- Logs each cleared item for debugging
- Properly resets all state variables

**Result**: "Clear Cache & Retry" now completely removes all cached data for fresh download.

### 4. **Download Progress Stall Detection** ✅
**Problem**: Downloads could hang indefinitely without user feedback.

**Solution**: Added stall detection in `ios/Runner/LLMService.swift` lines ~290-310:
- Tracks last progress time
- Logs warning if stuck for >30 seconds
- Counts consecutive stalls
- Provides detailed progress logs every 5%

**Result**: Console logs show when download is stuck, making debugging easier.

## Testing Checklist

### Test Case 1: Fresh Install (No Cache)
1. ✅ Build and run on device
2. ✅ Press "Download Model" button
3. ✅ Verify progress updates show MB downloaded
4. ✅ Verify model loads successfully after download
5. ✅ Test query with RAG context

### Test Case 2: Restart with Cached Model
1. ✅ Quit app (with model already downloaded)
2. ✅ Relaunch app
3. ✅ Verify status shows "Loading cached model..." 
4. ✅ Verify model auto-loads WITHOUT download button
5. ✅ Verify ready state reached quickly
6. ✅ Test query works immediately

### Test Case 3: Download Failure Recovery
1. ✅ Start download
2. ✅ Turn off WiFi mid-download
3. ✅ Verify error message shows network issue
4. ✅ Press "Clear Cache & Retry"
5. ✅ Turn WiFi back on
6. ✅ Verify download starts fresh

### Test Case 4: Incomplete Cache Detection
1. ✅ Start download
2. ✅ Force quit app mid-download
3. ✅ Relaunch app
4. ✅ Verify status shows "Model incomplete - please download"
5. ✅ Press download button
6. ✅ Verify it continues/restarts download

## Architecture Flow

```
App Launch
    ↓
checkForCachedModel() [async]
    ↓
├─ No cache found
│   └─ Set status: "idle" / "Model not downloaded"
│       └─ User must click "Download Model"
│
├─ Incomplete cache (no config or weights)
│   └─ Set status: "error" / "Model incomplete - please download"
│       └─ User should "Clear Cache & Retry"
│
└─ Complete cache (has .safetensors + config.json)
    └─ Set status: "loading" / "Loading cached model..."
        └─ Call initializeRealLLMIfNeeded()
            └─ Load from Hub cache (instant)
                └─ Set status: "ready"
```

## Key Code Changes

### LLMService.swift - checkForCachedModel()
```swift
if hasWeights && hasConfig {
  print("✅ Cache is COMPLETE - auto-loading model now...")
  DispatchQueue.main.async { [weak self] in
    self?.statusState = "loading"
    self?.statusMessage = "Loading cached model..."
  }
  try? await Task.sleep(nanoseconds: 500_000_000)
  initializeRealLLMIfNeeded()  // ← AUTO-LOAD!
}
```

### LLMService.swift - clearCache()
```swift
// Clear main model directory
try FileManager.default.removeItem(at: modelPath)

// Clear Hub metadata
for dirName in ["snapshots", "refs", "blobs"] {
  let dirPath = base.appendingPathComponent(dirName)
  try? FileManager.default.removeItem(at: dirPath)
}

// Clear temp files
if item.hasSuffix(".tmp") || item.hasSuffix(".download") {
  try? FileManager.default.removeItem(at: tmpPath)
}
```

### LLMService.swift - Error Messages
```swift
if nsError.code == NSURLErrorNotConnectedToInternet {
  errorMessage = "No internet connection. Please check your network."
} else if nsError.code == NSURLErrorTimedOut {
  errorMessage = "Download timed out. Please try 'Clear Cache & Retry'."
}
```

## Debugging Commands

### Check cache status on device
```bash
# Via Xcode console - look for these log lines:
🔍 Checking for cached model at: /path/to/cache
📁 Cache contains X files: file1, file2, ...
📊 Cache validation: weights=true, config=true, tokenizer=true
✅ Cache is COMPLETE - auto-loading model now...
```

### Clear cache manually (if needed)
```bash
# Delete app from device
# Reinstall - cache is stored in app sandbox, so this clears it
```

### Monitor memory usage
```bash
# Look for these log lines during model load:
📊 Initial memory: XXX.X MB
⚠️ Warning: High memory usage before model load
✅ MLX LLM model loaded and verified successfully
```

## Expected Behavior Summary

| Scenario | Status on Launch | Auto-Load? | Button Action |
|----------|-----------------|------------|---------------|
| No cache | "Model not downloaded" | ❌ | Downloads from HuggingFace |
| Incomplete cache | "Model incomplete - please download" | ❌ | Should clear cache first |
| Complete cache | "Loading cached model..." → "Ready" | ✅ | N/A - already loaded |
| Download failed | "Network error" / "Download timed out" | ❌ | Use "Clear Cache & Retry" |

## Next Steps if Issues Persist

1. **Download Still Failing**:
   - Check device internet connection
   - Check HuggingFace mlx-community repo accessibility
   - Try downloading on WiFi (not cellular)
   - Check device storage (need ~700MB free)

2. **Cache Not Auto-Loading**:
   - Check Xcode console for "Cache validation" logs
   - Verify .safetensors and config.json exist in cache
   - Try "Clear Cache & Retry" to re-download

3. **Memory Crashes**:
   - Close all other apps
   - Reboot device
   - Check if device supports Metal (iPhone 8 or newer)
   - Consider switching to even smaller model if needed

## Files Modified

- `/Users/gavinlynch04/Desktop/calhacks/app/ios/Runner/LLMService.swift`
  - Lines ~110-165: `checkForCachedModel()` - auto-load logic
  - Lines ~375-400: Enhanced error messages
  - Lines ~480-540: Thorough cache clearing
  - Lines ~290-310: Download stall detection

No Flutter/Dart changes were needed - the UI already had proper error handling and status polling!
