using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML;

namespace ML.NET_test.Prediction
{
    public class Engine<TInput, TOutput>
        where TInput : class
        where TOutput : class, new()
    {
        private PredictionEngine<TInput, TOutput> _engine;
        public Engine(PredictionEngine<TInput, TOutput> engine)
        {
            _engine = engine;
        }

        //----------------------------------------------------------------------
        public static Engine<TInput, TOutput> Create(Settings configuration, int tokens)
        {
            var context = new MLContext();

            ITransformer transformer = CreateModel(context, configuration, tokens);

            var engine = context.Model.CreatePredictionEngine<TInput, TOutput>(transformer);

            return new Engine<TInput, TOutput>(engine);
        }

        private static ITransformer CreateModel(MLContext context, Settings configuration, int tokens)
        {
            bool hasGpu = configuration.gpuEnabled;

            var dataView = context.Data.LoadFromEnumerable(new List<TInput>());

            var pipeline = context.Transforms
                .ApplyOnnxModel(
                    modelFile: configuration.ModelPath,
                    shapeDictionary: new Dictionary<string, int[]>
                    {
                        { "input_ids", new [] { 1, configuration.SequenceLength } },
                        { "token_type_ids", new [] { 1, configuration.SequenceLength } },
                        { "attention_mask", new [] { 1, configuration.SequenceLength } },
                    },
                    inputColumnNames: configuration.ModelInput,
                    outputColumnNames: configuration.ModelOutput,
                    gpuDeviceId: hasGpu ? 0 : (int?)null,
                    fallbackToCpu: true
                );

            var transformer = pipeline.Fit(dataView);

            return transformer;
        }

        //----------------------------------------------------------------------
        public TOutput Predict(TInput feature)
        {
            var result = _engine.Predict(feature);

            return result;
        }
    }
}
