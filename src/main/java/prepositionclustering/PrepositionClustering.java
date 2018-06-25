package prepositionclustering;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import it.uniroma2.sag.kelp.data.clustering.Cluster;
import it.uniroma2.sag.kelp.data.clustering.ClusterExample;
import it.uniroma2.sag.kelp.data.clustering.ClusterList;
import it.uniroma2.sag.kelp.data.dataset.SimpleDataset;
import it.uniroma2.sag.kelp.data.example.Example;
import it.uniroma2.sag.kelp.data.example.ExampleFactory;
import it.uniroma2.sag.kelp.data.manipulator.VectorConcatenationManipulator;
import it.uniroma2.sag.kelp.data.representation.Vector;
import it.uniroma2.sag.kelp.learningalgorithm.clustering.kmeans.LinearKMeansEngine;
import it.uniroma2.sag.kelp.learningalgorithm.clustering.kmeans.LinearKMeansExample;
import it.uniroma2.sag.kelp.utils.evaluation.ClusteringEvaluator;
import it.uniroma2.sag.kelp.wordspace.Wordspace;

public class PrepositionClustering {

	public static void main(String[] args) throws Exception {
		// load the embeddings 
		String matrixPath = "embeddings/verb_clustering_output_data_w3_f100_b20k_split_norm_250.txt";
		Wordspace ws = new Wordspace(matrixPath);
		
		String outputPath = "clusters.txt";
		OutputStream out = it.uniroma2.sag.kelp.utils.FileUtils.createOutputStream(outputPath);
		BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(out));
		
		String evalPath = "cluster_evaluation.txt";
		OutputStream outEval = it.uniroma2.sag.kelp.utils.FileUtils.createOutputStream(evalPath);
		BufferedWriter writerEval = new BufferedWriter(new OutputStreamWriter(outEval));
		
		writerEval.write("targetPreposition NumberOfExamples K purity mutualInformation nmi\n");
		
		// read the triples dataset
		String dataPath = "triples/streusle.tsv";
		InputStream in = it.uniroma2.sag.kelp.utils.FileUtils.createInputStream(dataPath);
		BufferedReader reader = new BufferedReader(new InputStreamReader(in));
		String line;
		SimpleDataset dataset = new SimpleDataset();
		Example ex = null;
		while ((line = reader.readLine()) != null) {
			String head = line.split("\t")[0];
			String headPos = line.split("\t")[1];
			String preposition = line.split("\t")[2];
			String modifier = line.split("\t")[3];
			String modifierPos = line.split("\t")[4];
			String dependency = line.split("\t")[5];
			String label = line.split("\t")[6];
			String representationDependency = "|BV:dependency| "+dependency+":1.0 |EV|";
			String representationPreposition = "|BV:preposition| "+preposition+":1.0 |EV|";
			String representationHeadPos = "|BV:headPos| "+headPos+":1.0 |EV|";
			String representationModifierPos = "|BV:modifierPos| "+modifierPos+":1.0 |EV|";
            Vector vectorHead = ws.getVector(head);
            Vector vectorModifier = ws.getVector(modifier);
            Vector vectorPreposition = ws.getVector(preposition+"::i");
            //if (vectorHead != null && vectorModifier != null  && vectorPreposition != null && preposition.equals(targetPreposition)) {
            if (vectorHead != null && vectorModifier != null  && vectorPreposition != null) {
    	            	//String instance = label + "|BS:info|"+head+" "+modifier+"|ES|"+ " |BDV:ws| " + vectorHead.toString() + "," + vectorModifier.toString() + " |EDV|"; 
            	String representationHeadVector = "|BDV:headVector| " + vectorHead.toString() + " |EDV|";
            	String representationModifierVector = "|BDV:modifierVector| " + vectorModifier.toString() + " |EDV|";
            	String representationPrepositionVector = "|BDV:prepositionVector| " + vectorPreposition.toString() + " |EDV|";
            	List<String> representations = new ArrayList<String>();
            	List<Float> weights = new ArrayList<Float>();
            	representations.add("headVector");
            	weights.add((float) 1.0);
            	representations.add("modifierVector");
            	weights.add((float) 1.0);
            	//representations.add("prepositionVector");
            	//weights.add((float) 1.0);
            	//representations.add("preposition");
            	//weights.add((float) 1.0);
            	representations.add("headPos");
            	weights.add((float) 1.0);
            	representations.add("modifierPos");
            	weights.add((float) 1.0);
            	representations.add("dependency");
            	weights.add((float) 1.0);
            	
            	//ex = ExampleFactory.parseExample(label + "|BS:info|"+head+" "+preposition+" "+modifier+" "+dependency+"|ES|"+representationHeadVector+representationModifierVector+representationPrepositionVector+representationPreposition+representationHeadPos+representationModifierPos+representationDependency);
            	ex = ExampleFactory.parseExample(label + "|BS:info|"+head+" "+preposition+" "+modifier+" "+dependency+"|ES|"+representationHeadVector+representationModifierVector+representationHeadPos+representationModifierPos+representationDependency);
            	VectorConcatenationManipulator concatenationManipulator = new VectorConcatenationManipulator("all",	representations,weights);
            	concatenationManipulator.manipulate(ex);
            	dataset.addExample(ex);
            	//System.out.println(ex);
            }
		}
			
		// clustering
		int K = dataset.getClassificationLabels().size();
		int tMax = 10;
		String representationName = "all";
		LinearKMeansEngine clusteringEngine = new LinearKMeansEngine(representationName, K, tMax);
		ClusterList clusterList = clusteringEngine.cluster(dataset);

		// Writing the resulting clusters and cluster members
		for (Cluster cluster : clusterList) {
			for (ClusterExample clusterMember : cluster.getExamples()) {
				float dist = ((LinearKMeansExample) clusterMember).getDist();
				Example e = clusterMember.getExample();
				//writer.write(targetPreposition + "\t" + dist + "\t" + cluster.getLabel() + "\t" + e.getLabels()[0] + "\t" + e.getRepresentation("info") + "\n");
				writer.write(e.getRepresentation("info") + "\t" + dist + "\t" + cluster.getLabel() + "\t" + e.getLabels()[0] + "\n");
			}
			writer.write("\n");
		}
			
		float purity = ClusteringEvaluator.getPurity(clusterList);
		float mutualInformation = ClusteringEvaluator.getMI(clusterList);
		float nmi = ClusteringEvaluator.getNMI(clusterList);
		writerEval.write(dataset.getNumberOfExamples() + " " + 
                   K  + " " + 
                   purity  + " " + 
                   mutualInformation  + " " + 
                   nmi + "\n");
		//System.out.println(ClusteringEvaluator.getStatistics(clusterList));

		writer.close();
		writerEval.close();
	}

}
