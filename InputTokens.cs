using Microsoft.ML.Data;

namespace ML.NET_test
{
    public class InputTokens
    {
        [VectorType(1, 256)]
        [ColumnName("input_ids")]
        public long[] Tokens { get; set; }

        [VectorType(1, 256)]
        [ColumnName("token_type_ids")]
        public long[] Token_type_ids { get; set; }

        [VectorType(1, 256)]
        [ColumnName("attention_mask")]
        public long[] Attention { get; set; }
    }
}
