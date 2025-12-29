# GPU Acceleration in argus-click

## Summary

Argus-click now uses hardware OpenGL by default (was forcing software rendering). This fixes lag issues for users with GPUs while remaining compatible with systems that don't have dedicated GPUs.

## If You Experience Issues

If you encounter display problems or crashes (rare, typically due to very old/buggy drivers), you can disable hardware acceleration:

### Windows
```cmd
set ARGUS_GPU_ACCELERATION=disable
argus-click
```

### macOS/Linux  
```bash
export ARGUS_GPU_ACCELERATION=disable
argus-click
```

This enables software rendering behavior.
