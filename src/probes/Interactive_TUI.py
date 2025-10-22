import sys
from pathlib import Path

# Add parent directory to path when running as script
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.probes.streaming_probe_inference import StreamingProbeInferenceEngine
from src.probes.interactive_probe_viewer import launch_interactive_viewer

engine = StreamingProbeInferenceEngine(
    probes_base_dir=Path('data/probes_binary'),
    model_name='google/gemma-3-4b-it',
    sentiment_probes_dir=Path('data/sentiment'),
    include_sentiment=True,
    layer_range=(15, 30)  # Use all available layers
)

# text = "I was in such a dark place back then, everything felt hopeless and I couldn't see any way forwardw. The anxiety was crushing me daily and I kept spiraling into these negative thought patterns where I'd convince myself that nothing would ever get better. I remember lying awake at night just replaying all my failures and mistakes over and over, feeling like I was trapped in this endless cycle of self-doubt and despair. But then something shifted when I started reaching out for help instead of isolating myself. I began talking to friends, seeking therapy, and slowly learning to challenge those destructive thoughts that had been controlling my life. It wasn't easy at first - there were setbacks and days when I felt like I was back at square one. But I kept pushing forward, developing new coping strategies, practicing mindfulness, and gradually rebuilding my confidence. I started setting small achievable goals and celebrating tiny victories instead of focusing on everything that was wrong. The breakthrough came when I realized I had the power to change my perspective and that my thoughts didn't define my reality. Now I'm in such a better place mentally, surrounded by supportive people, pursuing goals that actually matter to me, and I've learned that even in the darkest moments there's always hope if you're willing to take that first step toward healing."
threshold = 0.5
text = "I was in such a dark place back then, everything felt hopeless and I couldn't see any way forward. "



# Get ALL predictions (not just top_k) for aggregation
# Include both cognitive and sentiment probes in the count
total_probes = len(engine.probes) + len(engine.sentiment_probes)
all_predictions = engine.predict_streaming(
    text,
    top_k=total_probes,  # Get all predictions (cognitive + sentiment)
    threshold=0.0,  # Get all, filter during aggregation
    show_realtime=False
)

# Aggregate predictions by action across layers
aggregated_predictions = engine.aggregate_predictions(all_predictions, threshold=threshold)

# Sort by layer count and confidence
aggregated_predictions.sort(key=lambda x: (x.layer_count, x.max_confidence), reverse=True)

# Get tokens
inputs = engine.tokenizer(text, return_tensors="pt")
tokens = engine.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

# Launch viewer with correct parameters
launch_interactive_viewer(
    aggregated_predictions,
    tokens,
    (engine.layer_start, engine.layer_end),  # layer_range tuple
    threshold  # display_threshold
)