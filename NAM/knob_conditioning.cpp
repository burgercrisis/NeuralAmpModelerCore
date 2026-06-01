#include "knob_conditioning.h"

#include <cmath>
#include <cstring>
#include <sstream>
#include <stdexcept>

#include "dsp.h"

#define tanh_impl_ std::tanh

// =============================================================================
// KnobConditioningDSP
// =============================================================================
//
// Computes per-knob embeddings for FiLM conditioning in a multi-knob WaveNet.
//
// Input: 1 channel (audio passthrough from parent WaveNet, unused).
// Output: K * embedding_dim channels, where K = number of knobs.
//
// For each knob k with value v_k, the embedding is:
//   embedding[k][i] = weight[k][i] * v_k + bias[k][i]
//   for i = 0 .. embedding_dim-1
//
// The same embedding vector is applied identically across all time steps.
//
// Weight layout (flat):
//   Per knob k: [weight(embedding_dim floats), bias(embedding_dim floats)]
//   Total: knob_count * embedding_dim * 2
//
// Thread safety: SetExternalInputs() is called from the UI thread, process()
// from the audio thread. Each knob value is a single aligned float which is
// atomic on x86/x64. The process() function copies all knob values to a local
// stack array before using them, so partial updates during processing are benign.

namespace nam
{
namespace knob_conditioning
{

class KnobConditioningDSP : public DSP
{
private:
   std::vector<std::string> mKnobNames;
   int mEmbeddingDim;
   int mKnobCount;

   // Flat weight storage: [knob_0: weight[emb_dim], bias[emb_dim], knob_1: ...]
   std::vector<float> mWeights;

   // Per-knob values — written by UI thread (SetExternalInputs),
   // read by audio thread (process()). Aligned float reads are atomic on x86/x64.
   std::vector<float> mKnobValues;

public:
   KnobConditioningDSP(const std::vector<std::string>& knob_names, int embedding_dim,
                       const std::vector<float>& weights, double expected_sample_rate)
   : DSP(1, (int)knob_names.size() * embedding_dim, expected_sample_rate)
   , mKnobNames(knob_names)
   , mEmbeddingDim(embedding_dim)
   , mKnobCount((int)knob_names.size())
   , mKnobValues((size_t)mKnobCount, 0.0f)
   {
      const size_t expected = (size_t)knob_names.size() * (size_t)embedding_dim * 2;
      if (weights.size() != expected)
      {
         std::stringstream ss;
         ss << "KnobConditioningDSP: expected " << expected << " weights, got " << weights.size();
         throw std::runtime_error(ss.str());
      }
      mWeights = weights;
   }

   void SetExternalInputs(const std::vector<float>& params) override
   {
      if ((int)params.size() != mKnobCount)
      {
         std::stringstream ss;
         ss << "KnobConditioningDSP::SetExternalInputs: expected " << mKnobCount << " params, got "
            << params.size();
         throw std::runtime_error(ss.str());
      }
      for (int i = 0; i < mKnobCount; i++)
         mKnobValues[i] = params[i];
      // Also store in base class vector for generic access
      mExternalInputs = params;
   }

   void process(NAM_SAMPLE** input, NAM_SAMPLE** output, const int num_frames) override
   {
      // Copy knob values to local stack array for audio-thread-local use.
      // On x86/x64, aligned float reads are atomic, so this is safe for
      // single-writer (UI thread) / single-reader (audio thread) scenarios.
      float knob_values[32];
      float* kv_ptr;
      std::vector<float> kv_heap;

      if (mKnobCount <= 32)
      {
         kv_ptr = knob_values;
      }
      else
      {
         // Unlikely path: more than 32 knobs
         kv_heap.resize(mKnobCount);
         kv_ptr = kv_heap.data();
      }

      for (int k = 0; k < mKnobCount; k++)
         kv_ptr[k] = mKnobValues[k];

      const int out_channels = NumOutputChannels();

      // Zero output first
      for (int ch = 0; ch < out_channels; ch++)
         for (int s = 0; s < num_frames; s++)
            output[ch][s] = (NAM_SAMPLE)0.0;

      // For each time step, compute embedding: output[ch][s] = w * kv + b
      // This is identical across all time steps since knob values don't change per-sample
      for (int s = 0; s < num_frames; s++)
      {
         for (int k = 0; k < mKnobCount; k++)
         {
            const float kv = kv_ptr[k];
            const int w_offset = k * mEmbeddingDim * 2;
            const int out_ch_base = k * mEmbeddingDim;
            for (int i = 0; i < mEmbeddingDim; i++)
            {
               const float w = mWeights[w_offset + i];
               const float b = mWeights[w_offset + mEmbeddingDim + i];
               output[out_ch_base + i][s] = (NAM_SAMPLE)(w * kv + b);
            }
         }
      }
   }

   int PrewarmSamples() override { return 0; }
};

// =============================================================================
// KnobConditioningConfig
// =============================================================================

KnobConditioningConfig::KnobConditioningConfig(const nlohmann::json& config, double sampleRate)
{
   (void)sampleRate;

   // Parse knob_names
   if (!config.contains("knob_names"))
      throw std::runtime_error("KnobConditioning config missing 'knob_names'");
   for (const auto& name : config["knob_names"])
      knob_names.push_back(name.get<std::string>());

   // Parse embedding_dim
   if (!config.contains("embedding_dim"))
      throw std::runtime_error("KnobConditioning config missing 'embedding_dim'");
   embedding_dim = config["embedding_dim"].get<int>();

   in_channels = 1; // Must match _get_condition_dim() of parent WaveNet
   out_channels = (int)knob_names.size() * embedding_dim;
}

std::unique_ptr<DSP> KnobConditioningConfig::create(std::vector<float> weights, double sampleRate)
{
   (void)sampleRate;
   return std::make_unique<KnobConditioningDSP>(knob_names, embedding_dim, weights,
                                                sampleRate >= 0 ? sampleRate : -1.0);
}

// =============================================================================
// Config parser + registration
// =============================================================================

std::unique_ptr<ModelConfig> create_config(const nlohmann::json& config, double sampleRate)
{
   auto cfg = std::make_unique<KnobConditioningConfig>();
   auto parsed = KnobConditioningConfig(config, sampleRate);
   *cfg = std::move(parsed);
   return cfg;
}

// Register the "KnobConditioning" architecture at program startup.
namespace
{
static nam::ConfigParserHelper _register_KnobConditioning("KnobConditioning", nam::knob_conditioning::create_config);
}

} // namespace knob_conditioning
} // namespace nam
