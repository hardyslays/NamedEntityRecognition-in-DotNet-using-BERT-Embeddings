// See https://aka.ms/new-console-template for more information

using ML.NET_test.Tokenizer;
using MoreLinq.Extensions;

namespace ML.NET_test
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Setting up the project...");
            var settings = new Settings();
            var tokenizer = WordTokenizer.FromVocabularyFile(settings.VocabPath);
            var config = Config.FromFile(settings.ConfigPath);
            Console.WriteLine("Project ready to go...\n");


            List<string> sentences = [
                "I am Raj and I work at Microsoft."
                ];

            var tokens = tokenizer.Tokenize(sentence.ToArray()).ToArray();
            Console.WriteLine($"[{string.Join(',', tokens)}]");

            var padded = tokens.Select(t => (long)t.VocabularyIndex).Concat(Enumerable.Repeat(0L, settings.SequenceLength - tokens.Length)).ToArray();

            var token_type_ids = Enumerable.Repeat(0L, padded.Length).ToArray();
            var attentionMask = Enumerable.Repeat(1L, padded.Length).ToArray();

            var feature = new InputTokens { Tokens = padded, Token_type_ids = token_type_ids, Attention = attentionMask };

            var engine = Prediction.Engine<InputTokens, Output>.Create(settings, padded.Length);

            var result = engine.Predict(feature);

            //Console.WriteLine(result.Result);
            //Console.WriteLine(tokens);
            //Convert logits to readable output
            tokens
                .Zip(result.Result.Batch(9).ToArray(), (token, values) => (Token: token, Values: values))
            .GroupBy(tuple => (WordIndex: tuple.Token.WordIndex, Word: tuple.Token.Word))
            .Select(group => GetWordCategory(config, group.Key.WordIndex, group.Key.Word, group.SelectMany(g => g.Values)))
            .Where(tuple => tuple.Category > 0)
            .ForEach(tuple => Console.WriteLine($"Word: {tuple.Word}, Label: {tuple.Label}"))
            ;


            //tmp.ForEach(t =>
            //{
            //    Console.WriteLine(t.Token + ": ");
            //    t.Values.ForEach(v => Console.Write(v + " "));
            //    Console.WriteLine();
            //});

            //tmp.ForEach(t => {
            //    Console.WriteLine(t.ToString());
            //});
        }
        private static (int WordIndex, string Word, int Category, string Label, float Score) GetWordCategory(Config config, int wordIndex, string word, IEnumerable<float> values)
        {
            return values
                .Select((v, i) => (Value: v, Index: i))
                .GroupBy(values => values.Index % 9)
                .Select((group, index) => (Category: index, Value: group.Average(g => g.Value)))
                .Where(tuple => tuple.Value > 0.1)
                .OrderByDescending(tuple => tuple.Value)
                .Select(tuple => (wordIndex, word, tuple.Category, config.id2label[tuple.Category.ToString()], tuple.Value))
                .FirstOrDefault();
            ;
        }
    }
}