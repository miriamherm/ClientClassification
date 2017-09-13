package com.mycompany.app;

import edu.stanford.nlp.ie.AbstractSequenceClassifier;
import edu.stanford.nlp.ie.crf.*;
import edu.stanford.nlp.ling.CoreLabel;

import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;




import java.util.Set;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVRecord;

/** This is a demo of calling CRFClassifier programmatically.
 *  <p>
 *  Usage: {@code java -mx400m -cp "*" NERDemo [serializedClassifier [fileName]] }
 *  <p>
 *  If arguments aren't specified, they default to
 *  classifiers/english.all.3class.distsim.crf.ser.gz and some hardcoded sample text.
 *  If run with arguments, it shows some of the ways to get k-best labelings and
 *  probabilities out with CRFClassifier. If run without arguments, it shows some of
 *  the alternative output formats that you can get.
 *  <p>
 *  To use CRFClassifier from the command line:
 *  </p><blockquote>
 *  {@code java -mx400m edu.stanford.nlp.ie.crf.CRFClassifier -loadClassifier [classifier] -textFile [file] }
 *  </blockquote><p>
 *  Or if the file is already tokenized and one word per line, perhaps in
 *  a tab-separated value format with extra columns for part-of-speech tag,
 *  etc., use the version below (note the 's' instead of the 'x'):
 *  </p><blockquote>
 *  {@code java -mx400m edu.stanford.nlp.ie.crf.CRFClassifier -loadClassifier [classifier] -testFile [file] }
 *  </blockquote>
 *
 *  @author Jenny Finkel
 *  @author Christopher Manning
 */

public class StanfordNER 
{
    public static void main( String[] args )throws Exception 
    {
    	File file = new File("C:/Users/Miriam/Documents/MastersResearch/DataScience/DataSources/results_names_train.txt");
    	// creates the file
        file.createNewFile();
     // creates a FileWriter Object
        FileWriter writer = new FileWriter(file); 

        
    	String serializedClassifier = "C:/Program Files (x86)/stanford-ner-2017-06-09/classifiers/english.all.3class.distsim.crf.ser.gz";

     //   if (args.length > 0) {
     //     serializedClassifier = args[0];
     //   }

        AbstractSequenceClassifier<CoreLabel> classifier = CRFClassifier.getClassifier(serializedClassifier);

        /* For either a file to annotate or for the hardcoded text example, this
           demo file shows several ways to process the input, for teaching purposes.
        */

        	Reader in = new FileReader("C:/Users/Miriam/Documents/MastersResearch/DataScience/DataSources/train_names.txt");
    		Iterable<CSVRecord> records = CSVFormat.TDF.withFirstRecordAsHeader().parse(in);
    		
    		
    		Map<String, Map<String, Integer>> colsToEntities = new HashMap<String, Map<String, Integer>>(); 
    		try{
    		for (CSVRecord record : records) {
    		
    			Map<String, String> m = record.toMap();
    		    for (Map.Entry<String, String> entry : m.entrySet()) {
    		    	
    		    	String s = entry.getValue();
    		    	String[] example = s.split(" ");
    		    	String[] types;

    		    	String classified_line="";
    		    	List<String> list = new ArrayList<String>(); ;
    		    	
    		    	
    		    	//System.out.println("value:" + entry.getValue() + " has: " + s + " in column:" + entry.getKey());
    		    	 for (String str : example) {
    		    	     // Writes the content to the file
    		    		 String classified=classifier.classifyToString(str) +" ";
    		    		 classified_line+=classified;
    		    		 types=classified.split("/");
    		    		 list.add(types[types.length-1].trim());
    		    		 //System.out.print(types[types.length-1].trim());
    		           }
    		    
    		    	
    		    	 Set<String> set = new HashSet<String>(list);

    		    	 if (set.size()>1){
    		    		 classified_line+="\t mixed";
    		    	 }
    		    	 else{
    		    		 classified_line+="\t "+ set.iterator().next();
    		    	 }
    		    	 
    		    	 writer.write(classified_line + "\n"); 
    		    	
    		    }
    		
    			
    			}
    	
    		writer.flush();
            writer.close();
    		}
	    	catch (Exception e){
	    		System.out.println(e);
	    	}        
    		/*	
    		// pretty print
    		for (String k : colsToEntities.keySet()) {
    			System.out.println(k + colsToEntities.get(k));
    		}
    	}

          String[] example = {"Good afternoon Rajat Raina, how are you today?",
                              "I go to school at Stanford University, which is located in California." };
              for (String str : example) {
            System.out.print(classifier.classifyToString(str, "tsv", false));
          }
          System.out.println("---"); 
      
          */
          
          /*  
          for (String str : example) {
            System.out.println(classifier.classifyToString(str));
          }
          System.out.println("---");

          for (String str : example) {
            // This one puts in spaces and newlines between tokens, so just print not println.
            System.out.print(classifier.classifyToString(str, "slashTags", false));
          }
          System.out.println("---");

          for (String str : example) {
            // This one is best for dealing with the output as a TSV (tab-separated column) file.
            // The first column gives entities, the second their classes, and the third the remaining text in a document
            System.out.print(classifier.classifyToString(str, "tabbedEntities", false));
          }
          System.out.println("---");

          for (String str : example) {
            System.out.println(classifier.classifyWithInlineXML(str));
          }
          System.out.println("---");

          for (String str : example) {
            System.out.println(classifier.classifyToString(str, "xml", true));
          }
          System.out.println("---");

     

         // This gets out entities with character offsets
          int j = 0;
          for (String str : example) {
            j++;
            List<Triple<String,Integer,Integer>> triples = classifier.classifyToCharacterOffsets(str);
            for (Triple<String,Integer,Integer> trip : triples) {
              System.out.printf("%s over character offsets [%d, %d) in sentence %d.%n",
                      trip.first(), trip.second(), trip.third, j);
            }
          }
          System.out.println("---");

          // This prints out all the details of what is stored for each token
          int i=0;
          for (String str : example) {
            for (List<CoreLabel> lcl : classifier.classify(str)) {
              for (CoreLabel cl : lcl) {
                System.out.print(i++ + ": ");
                System.out.println(cl.toShorterString());
              }
            }
          }

          System.out.println("---");
*/
        
      }
}



