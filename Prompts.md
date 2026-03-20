## Pomegranate Patch
**Date:** 2026-03-19
**Prompt:** "pomegranate failed on macos because of some cython issue. Please find a fix for this."
**File changed:** abides_markets/models/order_size_model.py
**Solution:** Replaced pomegranate GeneralMixtureModel with numpy equivalent 
preserving identical LogNormal + Normal mixture components, weights, and parameters.