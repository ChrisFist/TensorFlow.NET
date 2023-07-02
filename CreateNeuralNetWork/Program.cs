using System;

namespace CreateNeuralNetWork
{
    internal class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Hello, World!");

            var trainingTF = new double[][]
          {
                 
                new double[] { 1, 1, 1, 0, 1, 1, 1, 0 },
                new double[] { 0, 0, 1, 0, 0, 0, 1, 1 },
                new double[] { 0, 1, 1, 1, 1, 1, 0, 2 },
                new double[] { 0, 1, 1, 1, 0, 1, 1, 3 },
                new double[] { 1, 0, 1, 1, 0, 0, 1, 4 },
                new double[] { 1, 1, 0, 1, 0, 1, 1, 5 },
                new double[] { 1, 1, 0, 1, 1, 1, 1, 6 },
                new double[] { 0, 1, 1, 0, 0, 0, 1, 7 },
                new double[] { 1, 1, 1, 1, 1, 1, 1, 8 },
                new double[] { 1, 1, 1, 1, 0, 0, 1, 9 },
               
          };

            NeuralNetworkGenerator NTF = new NeuralNetworkGenerator();
            trainingTF = NTF.GenerateTrainingData(trainingTF);
            //double[][] normalizedTrainingData = NTF.NormalizeData(trainingTF);

            //Console.WriteLine("Input Data before normalization:");
            //for (int i = 0; i < normalizedTrainingData.Length; i++)
            //{
            //    Console.WriteLine(string.Join(", ", trainingTF[i]));
            //}

            var graph = NTF.BuildNeuralNetwork(NeuralNetworkGenerator.NeuralNetworkType.distinct);
            NTF.TrainNeuralNetwork(graph, trainingTF, 1000);

            var testTF = new double[][]
            {
                new double[] { 0, 0, 1, 0, 0, 0, 1, 1 },
                new double[] { 0, 1, 1, 1, 1, 1, 0, 2 },
                new double[] { 0, 1, 1, 1, 0, 1, 1, 3 },
                new double[] { 1, 0, 1, 1, 0, 0, 1, 4 },
                new double[] { 1, 1, 0, 1, 0, 1, 1, 5 },
                new double[] { 1, 1, 0, 1, 1, 1, 1, 6 },
                new double[] { 0, 1, 1, 0, 0, 0, 1, 7 },
                new double[] { 1, 1, 1, 1, 1, 1, 1, 8 },
                new double[] { 1, 1, 1, 1, 0, 0, 1, 9 },
                new double[] { 1, 1, 1, 0, 1, 1, 1, 0 },
            };

            var output = NTF.PredictOutputs(graph, testTF);
        
            // Print output data
            Console.WriteLine("predicted output Data:");
            for (int i = 0; i < testTF.Length; i++)
            {
               Console.WriteLine(string.Join(", ", trainingTF[i]));
            }

            Console.WriteLine("Output: Probabilities:");
            Console.WriteLine("===================== ");
            for (int i = 0;i < testTF.Length; i++) {
                Console.WriteLine("i=" + i+1 + "-->output probability: " + output[i]);
            }



        }
    }
}