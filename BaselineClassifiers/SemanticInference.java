
package com.mycompany.app;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.InputStream;
import java.io.Reader;
import java.net.URL;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVRecord;

import opennlp.tools.namefind.NameFinderME;
import opennlp.tools.namefind.TokenNameFinderModel;
import opennlp.tools.tokenize.TokenizerME;
import opennlp.tools.tokenize.TokenizerModel;
import opennlp.tools.util.Span;

public class SemanticInference {

	private List<NameFinderME> detectors;
	
	private TokenizerME tokenizer;

	public SemanticInference() throws Exception {
		
		InputStream inputStreamTokenizer = this.getClass().getClassLoader().getResourceAsStream("en-token.bin");
		TokenizerModel tokenModel = new TokenizerModel(inputStreamTokenizer);
		// Instantiating the TokenizerME class
		tokenizer = new TokenizerME(tokenModel);

		// Loading the NER models
		detectors = new LinkedList<NameFinderME>();

		//detectors.add(new NameFinderME(init(this.getClass().getClassLoader().getResource("models/en-ner-money.bin").getPath())));
		detectors.add(new NameFinderME(init(this.getClass().getClassLoader().getResource("en-ner-location.bin").getPath())));
		//detectors.add(new NameFinderME(init(this.getClass().getClassLoader().getResource("models/en-ner-date.bin").getPath())));
		//detectors.add(new NameFinderME(init(this.getClass().getClassLoader().getResource("models/en-ner-time.bin").getPath())));
		//detectors.add(new NameFinderME(init(this.getClass().getClassLoader().getResource("models/en-ner-percentage.bin").getPath())));
		detectors.add(new NameFinderME(init(this.getClass().getClassLoader().getResource("en-ner-person.bin").getPath())));
		detectors.add(new NameFinderME(init(this.getClass().getClassLoader().getResource("en-ner-organization.bin").getPath())));
	}
	
	private TokenNameFinderModel init(String fileName) throws Exception {
		InputStream inputStreamNameFinder = new FileInputStream(fileName);
		TokenNameFinderModel model = new TokenNameFinderModel(inputStreamNameFinder);
		// Instantiating the NameFinderME class
		return model;
	}
	
	public Set<String> detect(String text) {
		String tokens[] = tokenizer.tokenize(text);
		Set<String> e = new HashSet<String>();

		for (NameFinderME detector : detectors) {
			Span nameSpans[] = detector.find(tokens);
			// Printing the spans of the locations in the sentence
			for (Span s : nameSpans) {
				e.add(s.getType());
			}
		}
		return e;
	}
	
	public static void main(String[] args) throws Exception {
		SemanticInference infer = new SemanticInference();
		
		File file = new File("C:/Users/Miriam/Documents/MastersResearch/DataScience/DataSources/openNLP_results_address_test.txt");
    	// creates the file
        file.createNewFile();
     // creates a FileWriter Object
        FileWriter writer = new FileWriter(file); 
		
		Reader in = new FileReader("C:/Users/Miriam/Documents/MastersResearch/DataScience/DataSources/src/test_address.txt");
		Iterable<CSVRecord> records = CSVFormat.TDF.withFirstRecordAsHeader().parse(in);
		
		Map<String, Map<String, Integer>> colsToEntities = new HashMap<String, Map<String, Integer>>(); 
		for (CSVRecord record : records) {
			Map<String, String> m = record.toMap();
		    for (Map.Entry<String, String> entry : m.entrySet()) {
		    	
		    	Set<String> s = infer.detect(entry.getValue());
		    	System.out.println("value:" + entry.getValue() + " has: " + s);
		    	writer.write(entry.getValue() + "\t" + s + "\n"); 
		    	if (!colsToEntities.containsKey(entry.getKey())) {
		    		colsToEntities.put(entry.getKey(), new HashMap<String, Integer>());
		    	}
		    	Map<String, Integer> counts = colsToEntities.get(entry.getKey());
		    	for (String k : s) {
		    		if (counts.containsKey(k)) {
		    			int z = counts.get(k) + 1;
		    			counts.put(k, z);
		    		} else {
		    			counts.put(k,  1);
		    		}
		    	}
		    }
		}
		
		// pretty print
		for (String k : colsToEntities.keySet()) {
			System.out.println(k + colsToEntities.get(k));
		}
		
		writer.flush();
        writer.close();
	}
}
