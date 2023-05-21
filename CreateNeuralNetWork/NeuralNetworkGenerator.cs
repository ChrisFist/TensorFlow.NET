﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using Tensorflow;
using static Tensorflow.Binding;


namespace CreateNeuralNetWork
{
    public class NeuralNetworkGenerator
    {
        private string savePath = "C:\\Users\\Admin\\Downloads";

        private Session session = null;

        public Graph BuildNeuralNetwork()
        {
            //Environment.SetEnvironmentVariable("TF_CPP_MIN_LOG_LEVEL", "2");
            tf.Context.log_device_placement(true);
            var graph = new Graph();
            session = tf.Session(graph);

            // Define the input layer
            var input = tf.placeholder(TF_DataType.TF_DOUBLE, new Shape(-1, 7), "input");

            // Define the trainable variables
            var weights1 = tf.Variable(tf.random.normal(new Shape(7, 10)), dtype: TF_DataType.TF_DOUBLE, name: "weights1", trainable: true);
            var biases1 = tf.Variable(tf.zeros(new Shape(10)), dtype: TF_DataType.TF_DOUBLE, name: "biases1", trainable: true);
            var weights2 = tf.Variable(tf.random.normal(new Shape(10, 1)), dtype: TF_DataType.TF_DOUBLE, name: "weights2", trainable: true);
            var biases2 = tf.Variable(tf.zeros(new Shape(1)), dtype: TF_DataType.TF_DOUBLE, name: "biases2", trainable: true);

            // Define the hidden layer
            var hidden = tf.add(tf.matmul(input, weights1), biases1);

            // Define the output layer
            var output = tf.add(tf.matmul(hidden, weights2), biases2, "label");

            // Apply the activation function to the output layer (e.g., sigmoid)
            var activatedOutput = tf.sigmoid(output, "activated_output");

            return graph;
        }

        public void TrainNeuralNetwork(Graph graph, double[][] trainingData, int numEpochs)
        {
            var trainingHistory = new TrainingHistory();
            // Prepare the training data
            var inputs = trainingData.Select(data => data.Take(data.Length - 1).ToArray()).ToArray();
            var labels = trainingData.Select(data => data[data.Length - 1]).ToArray();

            //var inputs = trainingData.Select(data => new Tensor(data.Take(data.Length - 1).ToArray())).ToArray();
            //var labels = trainingData.Select(data => new Tensor(data[data.Length - 1])).ToArray();

            // Initialize variables
            session.run(tf.global_variables_initializer());

            //foreach (var input in graph.get_operations()) {
            //   string s= input.name;
            //   string name = input.name;
            //}

            // Define placeholders for inputs and labels
            var inputTensor = graph.get_operation_by_name("input").output;
            var labelTensor = graph.get_operation_by_name("label").output;


            // Define the loss and optimizer
            var outputTensor = graph.get_operation_by_name("activated_output").output;
            var labelPlaceholder = tf.placeholder(TF_DataType.TF_DOUBLE, new Shape(-1));
            var loss = tf.reduce_mean(tf.square(tf.subtract(labelPlaceholder, outputTensor)));

            var optimizer = tf.train.GradientDescentOptimizer(learning_rate: 0.01f);
            var trainableVariables = graph.get_collection_ref<IVariableV1>("trainable_variables");
            Operation optimizerOp = null;
            if (trainableVariables != null)
            {
                optimizerOp = optimizer.minimize(loss, var_list: trainableVariables);

            }

            if (optimizerOp == null)
            {

                Console.WriteLine("Help");
            }

            //tf.train.export_meta_graph(@"C:\\Users\\Admin\\Downloads\\meta.txt", as_text: true);
            //var showgraph = graph.as_graph_def();
            //showgraph.WriteTo()

            // Perform training
            for (int epoch = 0; epoch < numEpochs; epoch++)
            {

                if (optimizerOp != null)
                {// Run one training step
                    Console.WriteLine("input");
                    var inp = new FeedItem(inputTensor, inputs);
                    print(inp);
                    Console.WriteLine("Labels");
                    var lab = new FeedItem(labelPlaceholder, labels);
                    print(lab);

                    var value = inp.Value as Array;  // Get the value of the feed item
                    Console.WriteLine(value);
                    value = lab.Value as Array;
                    Console.WriteLine(value);

                    session.run(optimizerOp, new FeedItem[] { new FeedItem(inputTensor, inputs), new FeedItem(labelPlaceholder, labels) });
                }
                // Calculate and store metrics
                var epochLoss = session.run(loss, new FeedItem(inputTensor, inputs), new FeedItem(labelPlaceholder, labels));
                trainingHistory.Loss.Add(epochLoss);
            }


            var saver = tf.train.Saver();
            saver.save(session, savePath);
            // Save training history to CSV
            SaveTrainingHistoryToCsv(trainingHistory, savePath);

        }

        public float[] PredictOutputs(Graph graph, double[][] inputData)
        {
            // Prepare the input data
            var inputs = inputData.Select(data => data.Take(data.Length - 1).ToArray()).ToArray();


            // Restore the saved model variables
            var saver = tf.train.Saver();
            saver.restore(session, savePath);

            // Get the input and output tensors
            var inputTensor = graph.get_operation_by_name("input").outputs[0];
            var outputTensor = graph.get_operation_by_name("activated_output").outputs[0];

            // Perform inference
            var outputs = session.run(outputTensor, new FeedItem(inputTensor, inputs));

            // Convert the output to float array
            var predictedOutputs = outputs.ToArray<float>();

            return predictedOutputs;

        }

        private void SaveTrainingHistoryToCsv(TrainingHistory trainingHistory, string savePath)
        {
            using (var writer = new StreamWriter(savePath))
            {
                // Write header
                writer.WriteLine("Epoch,Loss");

                // Write data rows
                for (int epoch = 0; epoch < trainingHistory.Loss.Count; epoch++)
                {
                    writer.WriteLine($"{epoch + 1},{trainingHistory.Loss[epoch]}");
                }
            }
        }

        public double[][] GenerateTrainingData(double[][] training)
        {
            List<double[]> newCombinations = new List<double[]>();

            // Generate all possible combinations of 7-digit states
            for (int i = 0; i < 128; i++)
            {
                string binaryString = Convert.ToString(i, 2).PadLeft(7, '0'); // Convert to binary string with leading zeros
                double[] combination = binaryString.Select(c => double.Parse(c.ToString())).ToArray();

                // Check if the combination is already present in the training array
                bool exists = training.Any(t => t.Take(7).SequenceEqual(combination));

                if (!exists)
                {
                    double[] newCombination = new double[8];
                    Array.Copy(combination, newCombination, 7);
                    newCombination[7] = -1; // Assign an output value of -1 for the new combination
                    newCombinations.Add(newCombination);
                }
            }

            // Add the new combinations to the training array
            double[][] updatedTraining = training.Concat(newCombinations).ToArray();

            return updatedTraining;
        }
    }

    public class TrainingHistory
    {
        public List<double> Loss { get; set; }

        public TrainingHistory()
        {
            Loss = new List<double>();
        }
    }


}
