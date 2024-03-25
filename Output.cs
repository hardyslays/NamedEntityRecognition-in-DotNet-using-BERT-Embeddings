using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML.Data;

namespace ML.NET_test
{
    public class Output
    {
        [VectorType(1, 256)]
        [ColumnName("logits")]
        public float[] Result { get; set; }
    }
}
