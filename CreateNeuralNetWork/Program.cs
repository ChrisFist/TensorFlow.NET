namespace CreateNeuralNetWork
{
    internal class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Hello, World!");

            var trainingTF = new double[][]
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

            NeuralNetworkGenerator NTF = new NeuralNetworkGenerator();
            //trainingTF = NTF.GenerateTrainingData(trainingTF);
            var graph = NTF.BuildNeuralNetwork();
            NTF.TrainNeuralNetwork(graph, trainingTF, 500);
            var output = NTF.PredictOutputs(graph, trainingTF);
        }
    }
}