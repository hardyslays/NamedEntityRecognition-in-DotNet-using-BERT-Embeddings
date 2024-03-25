using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;

namespace ML.NET_test
{
    public class Settings
    {
        private static string? Folder { get; set; } = Directory.GetParent(Environment.CurrentDirectory).Parent.Parent.FullName;
        public string ModelPath { get; set; } = Path.Join(Folder, "/Assets/model.onnx");
        // public string ModelPath { get; set; } =  "./Assets/distilbert-base-cased-finetuned-conll03-english.onnx"; 

        public string ConfigPath { get; set; } = Path.Join(Folder, "/Assets/config.json");

        public string VocabPath { get; set; } = Path.Join(Folder, "/Assets/vocab.txt");

        public string[] ModelInput => new[] { "input_ids", "token_type_ids", "attention_mask" };

        public string[] ModelOutput => new[] { "logits" };

        public int SequenceLength => 256;

        public bool gpuEnabled => false;
    }
}
