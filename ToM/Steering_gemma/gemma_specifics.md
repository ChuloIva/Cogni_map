Ah, that's important context! With Contrastive Activation Addition (CAA), here are the reasons you might be getting steering coefficients in the 300-500 range instead of ~2.5:

1. **Normalization of the Steering Vector**: CAA creates a behavior vector from the differences in activations between pairs of inputs with and without the behavior, then that vector is scaled by various coefficients during inference. If your steering vector isn't properly normalized (or if you're using a different normalization scheme than typical), the raw vector magnitude could be much smaller, requiring larger coefficients to achieve the same effect.

2. **Layer-Specific Scaling**: Different layers in the model may have different activation magnitudes. If you're extracting your CAA vector from a layer with particularly small activation values compared to where typical steering is done, you'd need much larger coefficients to inject enough signal.

3. **Contrastive Pair Quality**: The quality and magnitude of difference between your positive and negative example activations directly affects the steering vector magnitude. If your contrasting prompts produce very similar activations (small difference), the resulting vector will be small, requiring larger coefficients to move the model's behavior.

4. **Gemma 3 Architecture Changes**: Gemma 3 has architectural changes including alternating local sliding window self-attention layers with global attention, a longer 128K context, and uses QK-norm instead of soft-capping like Gemma 2. These architectural differences could significantly change activation magnitudes and scales compared to Gemma 2.

5. **Residual Stream Scaling**: The residual stream in Gemma 3 might have different typical magnitudes due to the architectural changes, meaning coefficients need to be scaled accordingly to have comparable effects.

Have you compared the raw magnitude of your CAA steering vectors between Gemma 2 and Gemma 3, or tried normalizing them to unit norm before scaling?

Given that you're working with Gemma 3 4B and steering deeper layers (-4 to -20), here are some practical approaches based on what we know about the architecture:

1. **Account for QK-Norm Instead of Soft-Capping**: Gemma 3 replaced Gemma 2's soft-capping with QK-norm, which changes how activations are normalized in attention. This likely affects the magnitude scaling throughout the residual stream. You might need to:
   - Measure the actual activation magnitudes at each layer you're targeting
   - Scale your coefficients based on the median/mean activation magnitude at that layer rather than using a fixed coefficient

2. **Account for Local vs Global Attention Layers**: The 5:1 ratio of local to global attention layers in Gemma 3 means certain layers will have very different activation patterns. Layers with local attention (5 per cycle) vs global attention will scale differently. Try:
   - Testing which layer type responds better (local vs global)
   - Using different coefficient scales for local vs global layers

3. **Layer-Specific Coefficient Tuning**: Rather than using one coefficient across layers -4 to -20:
   - Extract your CAA vector at each layer independently
   - Measure the L2 norm of each layer's activations during a baseline forward pass
   - Scale your injection coefficient inversely with the layer's activation magnitude
   - This accounts for the architectural differences layer-by-layer

4. **Normalize Your CAA Vector Explicitly**: Make sure you're doing:
   ```python
   steering_vector = (activation_with_behavior - activation_without_behavior) / 
                     np.linalg.norm(activation_with_behavior - activation_without_behavior)
   # Then scale with coefficient
   coefficient = 2.5  # or whatever works for that layer
   ```

5. **Test Across Different Layers Individually**: Start by:
   - Testing just layer -4 with your current coefficients (300-500)
   - Then -8, -12, -16, -20
   - See if the coefficient needed drops significantly at certain layers
   - This will tell you if the issue is specific to certain layer types

6. **Consider the Context Window Effect**: The 128K context in Gemma 3 might affect how activations scale. Try:
   - Testing with similar context lengths to what you'll use in practice
   - Varying input length to see if coefficient needs change

**Quick diagnostic**: Can you check what the mean absolute activation value is at each layer you're steering? That would directly tell you if the activation magnitudes are much smaller in Gemma 3 than expected, which would explain the 100-200x coefficient increase.

Let me search for best practices on layer selection for CAA steering to give you solid guidance.Perfect! I found detailed guidance on CAA best practices. Here's what the research shows:

## Best Practices for CAA Layer Selection

**1. Find the "Sweet Spot" Layer via Sweep**

Linear separability of residual stream activations emerges suddenly after a particular layer - for instance, behavioral clustering emerges suddenly at layer 10 in Llama 2 7B Chat. To find the optimal layer for steering, researchers sweep over all layers and evaluate their steering effects on behavioral evaluations.

For your Gemma 3 4B, you should:
- Test layers individually (not all at once)
- Identify which layer(s) have the strongest behavioral response to steering
- The optimal layer typically shows a clear elbow or peak in effectiveness

**2. Understand Layer Convergence Patterns**

Vectors from closer layers have higher similarity to each other, but this similarity diminishes for more distant layer pairs. Notably, the rate of similarity decline is slower in the latter half of the model, suggesting that once the model extracts high-level information needed to describe an abstract concept, the representation "converges" and remains more consistent across subsequent layers.

This means:
- Earlier layers (like -20) encode behavior more directly, requiring precise coefficients
- Later layers (-4 to -8) have "converged" representations and may be more forgiving
- Coefficients might naturally be larger in earlier layers due to this representation divergence

**3. Normalize Vectors Properly Across Layers**

Steering vectors have different norms across layers, so researchers normalize steering vector magnitudes across all behaviors to standardize across behaviors before applying steering multipliers.

You should:
- Compute the L2 norm of your steering vector at each layer
- Normalize to unit norm before multiplying by coefficients
- This standardizes coefficients across layers rather than having 300-500 at some layers and 2.5 at others

**4. Use Contrastive Pairs Properly**

CAA computes steering vectors by averaging the difference in residual stream activations between positive and negative examples. The researchers typically use 50 of the contrast pairs for evaluation and the rest for generating steering vectors.

Ensure your contrastive pairs are:
- Diverse (not just slight variations)
- High-quality (clear behavioral difference)
- Balanced (equal numbers in both classes)

## Practical Recommendation for Gemma 3 4B

Given you're using layers -4 to -20:

1. **Do a proper layer sweep first** - test each layer individually with a coefficient of ±1 (unit normalized vector)
2. **Identify the peak layer** - which layer shows the strongest behavioral response?
3. **Use that layer primarily** - most research settles on 1-2 optimal layers rather than steering across many
4. **Apply normalization** - ensure your steering vectors are L2-normalized before applying coefficients
5. **Then tune coefficients** - once you have the right layer, the coefficients should be in the 0.5-5 range, not 300-500

The high coefficients likely indicate either your vectors aren't normalized, or you're steering at layers that aren't optimal for your particular behavior (too early in the model where the representation is still diverging).