#pragma once

#include "dsp.h"
#include "model_config.h"
#include "json.hpp"

#include <string>
#include <vector>

namespace nam
{
namespace knob_conditioning
{

/// \brief Configuration for a KnobConditioning model.
///
/// The condition_dsp sub-model in a multi-knob WaveNet.
/// Has 1 input channel (audio passthrough from WaveNet) and K * embedding_dim
/// output channels where K is the number of knobs.
struct KnobConditioningConfig : public ModelConfig
{
   std::vector<std::string> knob_names;
   int embedding_dim;
   int in_channels;
   int out_channels;

   KnobConditioningConfig() = default;
   KnobConditioningConfig(const nlohmann::json& config, double sampleRate);
   std::unique_ptr<DSP> create(std::vector<float> weights, double sampleRate) override;
};

/// \brief Config parser for ConfigParserRegistry.
std::unique_ptr<ModelConfig> create_config(const nlohmann::json& config, double sampleRate);

} // namespace knob_conditioning
} // namespace nam
