package prepositionclassification;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

import it.uniroma2.sag.kelp.data.dataset.SimpleDataset;
import it.uniroma2.sag.kelp.data.example.Example;
import it.uniroma2.sag.kelp.data.example.ExampleFactory;
import it.uniroma2.sag.kelp.data.example.ParsingExampleException;
import it.uniroma2.sag.kelp.data.manipulator.VectorConcatenationManipulator;
import it.uniroma2.sag.kelp.data.representation.Vector;
import it.uniroma2.sag.kelp.data.representation.vector.SparseVector;
import it.uniroma2.sag.kelp.kernel.Kernel;
import it.uniroma2.sag.kelp.kernel.vector.LinearKernel;
import it.uniroma2.sag.kelp.learningalgorithm.classification.libsvm.BinaryCSvmClassification;
import it.uniroma2.sag.kelp.learningalgorithm.classification.multiclassification.OneVsAllLearning;
import it.uniroma2.sag.kelp.predictionfunction.classifier.ClassificationOutput;
import it.uniroma2.sag.kelp.predictionfunction.classifier.Classifier;
import it.uniroma2.sag.kelp.utils.JacksonSerializerWrapper;
import it.uniroma2.sag.kelp.utils.evaluation.MulticlassClassificationEvaluator;
import it.uniroma2.sag.kelp.wordspace.Wordspace;

public class PrepositionClassification {

	public static SimpleDataset readDataset(String dataPath, Wordspace ws, String subTree) throws IOException, InstantiationException, ParsingExampleException {
		InputStream in = it.uniroma2.sag.kelp.utils.FileUtils.createInputStream(dataPath);
		BufferedReader reader = new BufferedReader(new InputStreamReader(in));
		String line;
		SimpleDataset dataset = new SimpleDataset();
		Example ex = null;
		while ((line = reader.readLine()) != null) {
			String head = line.split("\t")[0];
			String preposition = line.split("\t")[1];
			String modifier = line.split("\t")[2];
			String role = line.split("\t")[4];
			String function = line.split("\t")[5];
			Vector vectorHead = ws.getVector(head);
            Vector vectorModifier = ws.getVector(modifier);
            Vector vectorPreposition = ws.getVector(preposition+"::i");
            if (vectorHead != null && vectorModifier != null  && vectorPreposition != null) {
            	String representationHeadVector = "|BDV:headVector| " + vectorHead.toString() + " |EDV|";
            	String representationModifierVector = "|BDV:modifierVector| " + vectorModifier.toString() + " |EDV|";
            	String representationPrepositionVector = "|BDV:prepositionVector| " + vectorPreposition.toString() + " |EDV|";
            	List<String> representations = new ArrayList<String>();
            	List<Float> weights = new ArrayList<Float>();
            	
            	if (subTree.equals("head")) {
            		ex = ExampleFactory.parseExample(role + "|BS:info|"+head+" "+preposition+"|ES|"+representationPrepositionVector +representationHeadVector);
            		representations.add("prepositionVector");
                	weights.add((float) 1.0);
                	representations.add("headVector");
                	weights.add((float) 1.0);
            	}
            	else if (subTree.equals("modifier")) {
            		ex = ExampleFactory.parseExample(role + "|BS:info|"+preposition+" "+modifier+"|ES|"+representationPrepositionVector+representationModifierVector);
            		representations.add("prepositionVector");
                	weights.add((float) 1.0);
                	representations.add("modifierVector");
                	weights.add((float) 1.0);
            	}
            	VectorConcatenationManipulator concatenationManipulator = new VectorConcatenationManipulator("all",	representations,weights);
            	concatenationManipulator.manipulate(ex);
            	//SparseVector v = VectorConcatenationManipulator.concatenateVectors(ex, representations, weights);
            	dataset.addExample(ex);
            }
		}
		return dataset;
	}
	
	public static void main(String[] args) throws Exception {
		// load the embeddings 
		String matrixPath = "embeddings/verb_clustering_output_data_w3_f100_b20k_split_norm_250.txt";
		Wordspace ws = new Wordspace(matrixPath);
		
		SimpleDataset datasetHTrain = readDataset("triples/streusle4.0_train.tsv", ws, "head");
		SimpleDataset datasetMTrain = readDataset("triples/streusle4.0_train.tsv", ws, "modifier");
		SimpleDataset datasetHDev = readDataset("triples/streusle4.0_dev.tsv", ws, "head");
		SimpleDataset datasetMDev = readDataset("triples/streusle4.0_dev.tsv", ws, "modifier");
		
		Kernel linearKernel = new LinearKernel("info");
		Kernel usedKernel = linearKernel;
		BinaryCSvmClassification svmSolver = new BinaryCSvmClassification();
		svmSolver.setKernel(usedKernel);
		float c = new Float(0.1);
		svmSolver.setCn(c);
		svmSolver.setCp(c);
		OneVsAllLearning ovaLearner = new OneVsAllLearning();
		ovaLearner.setBaseAlgorithm(svmSolver);
		
		System.out.println(datasetHTrain.getClassificationLabels());
		ovaLearner.setLabels(datasetHTrain.getClassificationLabels());
		JacksonSerializerWrapper serializer = new JacksonSerializerWrapper();
		serializer.writeValueOnFile(ovaLearner, "ova_head_learning_algorithm.klp");
		ovaLearner.learn(datasetHTrain);
		Classifier classifier = ovaLearner.getPredictionFunction();
		serializer.writeValueOnFile(classifier, "model_kernel-linear_cp.klp");

		MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator(datasetHTrain.getClassificationLabels());		
		for (Example e : datasetHDev.getExamples()) {
			// Predict the class
			ClassificationOutput p = classifier.predict(e);
			evaluator.addCount(e, p);
			System.out.println("Question:\t" + e.getRepresentation("info"));
			System.out.println("Original class:\t" + e.getClassificationLabels());
			System.out.println("Predicted class:\t" + p.getPredictedClasses());
			System.out.println();
		}

		System.out.println("Accuracy: " + evaluator.getAccuracy());
		
	}

}
