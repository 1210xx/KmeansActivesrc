package classification;

import java.io.FileReader;
import java.lang.reflect.Array;
import java.math.BigDecimal;
import java.util.*;
import weka.core.*;

public class RJ_CAKU003 {

	/**
	 * the instance status 0 untreated 1 brought 2 predicted
	 */
	int[] instanceStatus;

	/**
	 * the weka data
	 */
	Instances data;

	/**
	 * the center of cluster
	 */
	double[][] currentCenters;

	/**
	 * The obtained/predicted labels. To be compared with true labels.
	 */
	int[] labels;

	/**
	 * the misclassification cost setting
	 */
	double[] mCost;

	/**
	 * the teach cost setting
	 */
	double tCost;

	/**
	 **************************************
	 * The constructor.read the file and initialize the cost value
	 * 
	 * @param paraFilename   the data file
	 * @param paraMisclassificationCost the misclassification cost
	 * @param paraTeachCost  the teach cost
	 **************************************
	 */
	public RJ_CAKU003(String paraFilename, double[] paraMisclassificationCost, double paraTeachCost) {
		data = null;
		try {
			FileReader fileReader = new FileReader(paraFilename);
			data = new Instances(fileReader);
			fileReader.close();
			data.setClassIndex(data.numAttributes() - 1);
			// System.out.println("data.instance[data.numAttributes - 1]: " +
			// data.instance(1).value(data.numAttributes() - 1));
		} catch (Exception ee) {
			System.out.println("Cannot read the file: " + paraFilename + "\r\n" + ee);
			System.exit(0);
		} // Of try
		
		// Initialize
		mCost = paraMisclassificationCost;
		tCost = paraTeachCost;
		instanceStatus = new int[data.numInstances()];
		Arrays.fill(instanceStatus, 0);
		labels = new int[data.numInstances()];
		Arrays.fill(labels, -1);
		// System.out.println("(Construct)The data in constructor " + data);
		// System.out.println(data.instance(0).value(1));
	}

	/**
	 *************************************
	 * the main entrance
	 * 
	 * @param args system parameter
	 *************************************
	 */
	public static void main(String[] args) {
		// Test function call
		// testTotalCost();
		// testIndexToInstance();
		//testClassValue();
		double[] tempMisclassificationCost = { 2, 4 };
		double tempTeachCost = 1;
		double tempCost = 0;
		String dataTrace = "data/CAKU.arff";
		RJ_CAKU003 caku = new RJ_CAKU003(dataTrace, tempMisclassificationCost, tempTeachCost);

		// System.out.println("(main)Enter the main");
		// int[] tempCneterIndex = { 0, 20 };
		// System.out.println(Arrays.deepToString(caku.indexToInstance(tempCneterIndex)));
		// caku.cluster(caku.prelearn(), caku.indexToInstance(tempCneterIndex));
		caku.querySplitClassify(caku.prelearn(), 0);
		tempCost += caku.totalCost();
		
		//Detection the final result
		for (int i = 0; i < caku.prelearn().length; i++) {
			if (caku.instanceStatus[i] == 0) {
				System.out.println("The #" + i + "insntace untreat");
			}//of if
		}//of for i

		System.out.println("(main) The instance state is :  " + intArrayToString(caku.instanceStatus));
		System.out.println("(main) The label is :  " + intArrayToString(caku.labels));
		System.out.println("(main) the Cost is: " + tempCost);
		System.out.println("(main) Done....");
		System.out.println("(main) Read the data: " + dataTrace);
		System.out.println("(main) The data has " + caku.data.numInstances() + " instances");
	}// of main

	/**
	 ***********************************
	 * pre-learn make the WEKA data into integer array include the all data index
	 * 
	 * @return the processed array
	 ************************************
	 */
	public int[] prelearn() {
		
		int[] originalBlock = new int[data.numInstances()];
		for (int i = 0; i < originalBlock.length; i++) {
			originalBlock[i] = i;
		} // Of for i

		// System.out.println("(prelearn)instanceStatus: " +
		// Arrays.toString(instanceStatus));
		// System.out.println("(prelearn)labels: " + Arrays.toString(labels));
		return originalBlock;

	}// Of learningTest

	/**
	 ***********************************
	 * The main progress
	 * 
	 * @param paraBlock        the array to be processed
	 * @param paraInitialPoint the begin instances index in all block
	 ************************************
	 */
	public void querySplitClassify(int[] paraBlock, int paraInitialPoint) {
		int tempFirstLabel = -1;
		int tempFirstPoint = -1;
		System.out.println("(querySplitClassify) The instancestate of block is :" + intArrayToString(instanceStatus));
		System.out.println("(querySplitClassify) 1.The labels of block is :" + intArrayToString(labels));
		System.out.println("+++++++++++(querySplitClassify) The input block is :" + intArrayToString(paraBlock));
		
		//If block length < 5 buy it directly
		if (paraBlock.length <= 5) {
			for (int i = 0; i < paraBlock.length; i++) {
				instanceStatus[paraBlock[i]] = 1;
				labels[paraBlock[i]] = (int) data.instance(paraBlock[i]).classValue();
			} // Of for i
			return;
		}//of if
		
		//Judge the first point
		for (int i = 0; i < paraBlock.length; i++) {
			if (instanceStatus[paraBlock[i]] != 0) {
				tempFirstLabel = (int)data.instance(paraBlock[i]).classValue();
				System.out.println("Can i look the classvalue :" + (int)data.instance(paraBlock[i]).classValue());
				tempFirstPoint = paraBlock[i];
				break;
			}else {
				instanceStatus[paraInitialPoint] = 1;
				labels[paraInitialPoint] = (int)data.instance(paraInitialPoint).classValue();
				tempFirstLabel = (int)data.instance(paraInitialPoint).classValue();
				tempFirstPoint = paraInitialPoint;
			}//of else
		}//of for i
		
		System.out.println("(querySplitClassify) The first point is :" + tempFirstPoint);
		// count the queried instance of the block.
		int tempQueriedNum = 0;
		System.out.println("(querySplitClassify)  The  instance states  leght is : " + paraBlock.length);
		for (int i = 0; i < paraBlock.length; i++) {
			// System.out.println("(querySplitClassify) for i is : " + i);
			if (instanceStatus[paraBlock[i]] == 1) {
				tempQueriedNum++;
			} // of if
		} // of for i
		
		System.out.println("(querySplitClassify) The block " + tempQueriedNum + " instance has been brought");
		System.out.println("(querySplitClassify) The instancestate of block is :" + intArrayToString(instanceStatus));
		System.out.println("(querySplitClassify) 2. The labels of block is :" + intArrayToString(labels));
		// Step 1. Which instances have been queried in this block.
		int[] tempQueried = new int[tempQueriedNum];
		int tempIndex = -1;
		for (int i = 0; i < paraBlock.length; i++) {
			if (instanceStatus[paraBlock[i]] == 1) {
				++tempIndex;
				tempQueried[tempIndex] = paraBlock[i];
			} // of if
		} // of for i
		System.out.println("(querySplitClassify) The tempQueried array is : " + intArrayToString(tempQueried));

		// Step 2. How many instances to query
		// need to modify
		int tempNumToBuy = numToBuy(paraBlock, tempFirstLabel);

		System.out.println("(querySplitClassify) The " + tempNumToBuy + " instance need to buy");

		// Step 3. Set the tag of the block state

		boolean tempPure = true;

		// Step 4. Judge the block states . First check the block of obtained label
		tempQueried = new int[tempQueriedNum];
		tempIndex = -1;
		for (int i = 0; i < paraBlock.length; i++) {
			if (instanceStatus[paraBlock[i]] == 1) {
				++tempIndex;
				tempQueried[tempIndex] = paraBlock[i];
			} // of if
		} // of for j
		
		
		// Step 4.1. Query other instances one by one and update the second center
		int tempSecondCenter = -1;
		System.out.println("(querySplitClassify) Enter the buy one by one");
		for (int i = 0; i < tempNumToBuy; i++) {
			int tempFarthest = findFarthest(paraBlock, tempQueried);
			System.out.println("(querySplitClassify) The farthest instance is " + tempFarthest);
			// Query and update the states
			//System.out.println("(querySplitClassify) Buy the  " + tempFarthest + " instance");
			int tempCurrentLabel = (int) data.instance(tempFarthest).classValue();
			//System.out.println("(querySplitClassify) The curren label is " + tempCurrentLabel);
			labels[tempFarthest] = tempCurrentLabel;
			instanceStatus[tempFarthest] = 1;


			if (tempCurrentLabel != tempFirstLabel) {
				System.out.println("(querySplitClassify)  I need to quit and relearn");
				tempPure = false;
				tempSecondCenter = tempFarthest;
				break;
			}
			tempQueriedNum++;
			// update the farthest point
			tempQueried = new int[tempQueriedNum];
			tempIndex = -1;
			for (int j = 0; j < paraBlock.length; j++) {
				if (instanceStatus[paraBlock[j]] == 1) {
					++tempIndex;
					tempQueried[tempIndex] = paraBlock[j];
				} // of if
			} // of for j
			System.out.println(
					"(querySplitClassify) The tempQueried array after update is : " + intArrayToString(tempQueried));
			System.out.println("(querySplitClassify) Refind the farthest instance.......");

			tempFarthest = findFarthest(paraBlock, tempQueried);
			System.out.println("(querySplitClassify) The tempCurrentLabel is " + tempCurrentLabel
					+ "   and the tempFirstLael is : " + tempFirstLabel);

			// An instance with different label is queried.
			if (tempCurrentLabel != tempFirstLabel) {
				tempSecondCenter = tempFarthest;
				tempPure = false;
				break;
			} // of if

		} // Of for i

		// Step 6. Now split and re-learn
		if (!tempPure) {
			// make the two center to one array
			// Initialize
			System.out.println("(querySplitClassify) The block state is : " + tempPure);
			int[] tempCenterIndex = new int[2];
			tempCenterIndex[0] = tempFirstPoint;
			tempCenterIndex[1] = tempSecondCenter;
			System.out.println("(querySplitClassify) tempFirstCenter is " + tempCenterIndex[0]);
			System.out.println("(querySplitClassify) tempSecondCenter is " + tempCenterIndex[1]);
			double[][] tempCenter = new double[2][data.numAttributes()];
			tempCenter = indexToInstance(tempCenterIndex);
			System.out.println("(querySplitClassify) tempCenter is : " + intArrayToString(tempCenterIndex));
			int[][] tempSplitted = new int[2][paraBlock.length];
			System.out.println("(querySplitClassify) Getting splited ");
			tempSplitted = cluster(paraBlock, tempCenter);
			System.out.println("(quertempFirstPointySplitClassify) The cluster result  is : " + Arrays.deepToString(tempSplitted));
			
			System.out.println("(querySplitClassify) The Splitted1 initialPoint is: " + tempSplitted[0][0]);
			System.out.println("(querySplitClassify) The Splitted2 initialPoint is: " + tempSplitted[1][0]);

			querySplitClassify(tempSplitted[0], tempSplitted[0][0]);
			querySplitClassify(tempSplitted[1], tempSplitted[1][0]);

			} else {
			System.out.println("(querySplitClassify) I am get into prediceted");
			for (int i = 0; i < paraBlock.length; i++) {
				if (instanceStatus[paraBlock[i]] == 0) {
					labels[paraBlock[i]] = tempFirstLabel;
					instanceStatus[paraBlock[i]] = 2;
				} // of if
			} // of for i

			System.out.println(
					"(querySplitClassify) The instance states after prediceted : " + intArrayToString(instanceStatus));
			System.out.println("(querySplitClassify) 3.The labels after prediceted : " + intArrayToString(labels));
		} // of else
		System.out
				.println("+++++++++++++++++(querySplitClassify) The handle block is : " + intArrayToString(paraBlock));
		System.out.println("(querySplitClassify) 4.The labels after process is " + intArrayToString(labels));
		System.out.println(
				"(querySplitClassify) The instanceStatus after process is " + intArrayToString(instanceStatus));
	}// Of querySplitClassify

	/**
	 ***************
	 * Cluster. 2-Means Cluster split
	 * 
	 * @param paraBlock   The block to be processed
	 * @param paraCenters The 2 centers
	 ***************
	 */
	// need to modify
	public int[][] cluster(int[] paraBlock, double[][] paraCenters) {
		System.out.println("(cluster) I am gei in cluster");
		System.out.println("(cluster) The input center is : " + Arrays.deepToString(paraCenters));
		// Step 1. Initialize
		int paraK = 2;
		int[][] tempCluster = new int[paraK][paraBlock.length];
		int[][] resultCluster = null;
		double[][] tempNewCenters = paraCenters;
		double[][] tempCenters = new double[paraK][data.numAttributes() - 1];
		int[] tempKindSize = new int[paraK];

		// Step 2. Cluster and compute new centers.
		while (!doubleMatricesEqual(tempCenters, tempNewCenters)) {
			Arrays.fill(tempKindSize, 0);
			tempCenters = tempNewCenters;
			// Step 2.1 Cluster
			for (int i = 0; i < paraBlock.length; i++) {
				if (distance(paraBlock[i], tempCenters[0]) < distance(paraBlock[i], tempCenters[1])) {
					tempCluster[0][tempKindSize[0]] = paraBlock[i];
					tempKindSize[0]++; 
				} else {
					tempCluster[1][tempKindSize[1]] = paraBlock[i];
					tempKindSize[1]++;
				} // of else
			} // Of for i
			

			// Step 2.2 Sum all in one kind
			tempNewCenters = new double[paraK][data.numAttributes() - 1];
			for (int i = 0; i < paraK; i++) {
				// Sum
				for (int j = 0; j < tempKindSize[i]; j++) {
					for (int k = 0; k < data.numAttributes() - 1; k++) {
						tempNewCenters[i][k] += data.instance(tempCluster[i][j])
								.value(k);
					}// Of for k
				}// Of for j
			}//of for i
			
			// Step 2.3 Average Means conclude the new centers
			for (int i = 0; i < paraK; i++) {
				for (int j = 0; j < data.numAttributes() - 1; j++) {
					tempNewCenters[i][j] /= tempKindSize[i];
				} // Of for j
			} // Of for i
			
		} // Of while
		currentCenters = tempNewCenters;
		// Step 3. Let the cluster into 2-d array divide by label 
		resultCluster = new int[2][];
		for (int i = 0; i < paraK; i++) {
			resultCluster[i] = new int[tempKindSize[i]];
			for (int j = 0; j < tempKindSize[i]; j++) {
				resultCluster[i][j] = tempCluster[i][j];
			}// Of for j
		}// Of for i

		System.out.println("(cluster) All the block number: " + paraBlock.length + "\r(cluster) The 1 cluster number is "
				+ tempKindSize[0] + "\r(cluster) The 2 cluster number is " + tempKindSize[1]);
		System.out.println("(cluster)The result cluster is " + Arrays.deepToString(resultCluster));
		return resultCluster;
	}// Of cluster

	/**
	 ********************* 
	 * Look up optimal R and B
	 * 
	 * @param paraBlcokSize the input block size
	 ********************* 
	 */
	private int[] lookup(int paraBlcokSize) {
		// Linear Search
		// star[] :note the tmpMinCost instance in every query
		// starLoose: the final MinCost instance index
		// ra :the expect number of positive instances
		// ra[0] : compute the final starLoose as the compute the tmpCost
		// isFind : find the MinCost instance
		double[] tmpMinCost = new double[] { 0.5 * mCost[0] * paraBlcokSize, 0.5 * mCost[1] * paraBlcokSize };
		// double[] tmpMinCost = new double[] {mCost[0] * paraBlcokSize,
		// mCost[1] * paraBlcokSize }; //pair with the starLoose computation so
		// use the 0.5
		int[] star = new int[2];
		int[] starLoose = new int[2];
		boolean[] isFind = new boolean[2];
		double[] ra = new double[paraBlcokSize + 1];
		ra[0] = 0.5;
		double[] tmpCost = new double[2];
		for (int i = 1; i <= paraBlcokSize; i++) {
			if (paraBlcokSize >= 1000) {
				// the expect numbers of positive instances similarly the N approach infinite
				ra[i] = (i + 1.0) / (i + 2.0);
			} else {
				// else use the CADU equation 4 to compute the ra
				ra[i] = expectPosNum(i, 0, paraBlcokSize) / paraBlcokSize;
			}
			for (int j = 0; j < 2; j++) {
				tmpCost[j] = (1 - ra[i]) * mCost[j] * paraBlcokSize + tCost * i;
				if (tmpCost[j] < tmpMinCost[j]) {
					tmpMinCost[j] = tmpCost[j];
					star[j] = i;
					if (i == paraBlcokSize) {
						Arrays.fill(isFind, true);
					} // Of if
				} else {
					isFind[j] = true;
				} // Of if
			} // Of for j
				// System.out.println("(totalCost)star : " + intArrayToString(star));
				// System.out.println("(totalCost)cost :" + Arrays.toString(tmpCost));
				// System.out.println("(totalCost)QueriedInstance: " + i + ", cost: " +
				// tmpCost[0] + "\t");
			if (isFind[0] && isFind[1]) {
				Arrays.fill(isFind, false);
				for (int j = 0; j <= i; j++) {
					for (int k = 0; k < 2; k++) {
						tmpCost[k] = (1 - ra[j]) * mCost[k] * paraBlcokSize + tCost * j;
						if (tmpCost[k] <= tmpMinCost[k] && !isFind[k]) {
							isFind[k] = true;
							starLoose[k] = j;
						} // Of if
						if (isFind[0] && isFind[1]) {
							// System.out.println("(totalCost)------starLoose is : " +
							// intArrayToString(starLoose));
							return starLoose;
						} // Of if
					} // Of for k
				} // Of for j
			} // Of if
		} // Of for i
		return new int[] { 0, 0 };
		// throw new
		// RuntimeException("Error occured in lookup("+paraBlcokSize+")");
	}// Of lookup

	/**
	 ***********************************
	 * Compute the number to buy use the label Normal distribution
	 * 
	 * @param paraBlcok      THe block need to compute
	 * @param tempFirstLabel the first brought label
	 * @return the number need to buy
	 ************************************
	 */
	public int numToBuy(int[] paraBlock, int tempFirstLabel) {
		// lookup find optimal R B
		int[] tempNumBuys = lookup(paraBlock.length);
		int tempLabels = 0;
		for (int i = 0; i < paraBlock.length; i++) {
			if (instanceStatus[i] == 1) {
				tempLabels++;
			} // Of if
		} // Of for i

		int tempBuyLabels = 0;
		if (tempFirstLabel == 0 || tempFirstLabel == 1) {
			tempBuyLabels = tempNumBuys[tempFirstLabel] - tempLabels;
		} else {
			// tempFirstLabel = -1;
			tempBuyLabels = Math.max(tempNumBuys[0], tempNumBuys[1]) - tempLabels;
		} // Of if
		if (tempBuyLabels < 0) {
			tempBuyLabels = 0;
		}
		if (Math.sqrt(paraBlock.length) - tempBuyLabels > 0) {
			System.out.println("+");
		} else if (Math.sqrt(paraBlock.length) - tempBuyLabels < 0) {
			System.out.println("-");
		} // Of if else
		return tempBuyLabels;
	}// Of numToBuy

	/**
	 ************************* 
	 * Find the next farthest instance of the block.
	 * 
	 * @param paraBlock            The given block.
	 * @param paraLabeledInstances Labeled instances in the current block.
	 ************************* 
	 */
	public int findFarthest(int[] paraBlock, int[] paraLabeledInstances) {
		int resultFarthest = -1;
		double tempMaxDistanceSum = -1;
		for (int i = 0; i < paraBlock.length; i++) {
			double tempDistanceSum = 0;
			for (int j = 0; j < paraLabeledInstances.length; j++) {
				if (paraBlock[i] == paraLabeledInstances[j]) {
					tempDistanceSum = -1;
					// System.out.println("(findFarthest) Break.......");
					break;
				} // Of if
				tempDistanceSum += manhattanDistance(paraBlock[i], paraLabeledInstances[j]);
				// System.out.println("(findFarthest)" + paraBlock[i] + " to labeled " +
				// paraLabeledInstances[j] + " = " + manhattanDistance(paraBlock[i],
				// paraLabeledInstances[j]));
			} // Of for j
				// System.out.println("(findFarthest) The sum distance is " + tempDistanceSum +
				// " and the index is " + paraBlock[i]);
				// Update
			if (tempDistanceSum > tempMaxDistanceSum + 1e-6) {
				resultFarthest = paraBlock[i];
				tempMaxDistanceSum = tempDistanceSum;
			} // Of if
		} // Of for i
			// System.out.println("(findFarthest) The Max distance is " + tempMaxDistanceSum
			// + " and the index is " + resultFarthest);
		return resultFarthest;
	}// Of findFarthest

	/**
	 ***********************************
	 * Compute the Manhattan distance.
	 * 
	 * @param paraFirstIndex  the first instance index
	 * @param paraSecondIndex the second instance index
	 * @return the distance
	 ************************************
	 */
	public double manhattanDistance(int paraFirstIndex, int paraSecondIndex) {
		double tempResultDistance = 0;
		for (int i = 0; i < data.numAttributes() - 1; i++) {
			tempResultDistance += Math
					.abs(data.instance(paraFirstIndex).value(i) - data.instance(paraSecondIndex).value(i));
		} // Of for i

		return tempResultDistance;
	}// of mamhattanDistance

	/**
	 *****************************
	 * Compute Cost ----cost sensitive
	 * 
	 * @return the cost of one run
	 *****************************
	 */
	public double totalCost() {
		System.out.println("(totalCost)The instanceStatus in the totalCost " + intArrayToString(instanceStatus));
		System.out.println("(totalCost)The label in the totalCost " + intArrayToString(labels));
		double cost = 0;
		int tempTeachNum = 0;
		int temp1To0Num = 0;
		int temp0To1Num = 0;
		for (int i = 0; i < data.numInstances(); i++) {
			if (instanceStatus[i] == 1) {
				cost += tCost;
				tempTeachNum++;
			} else {
				if (labels[i] == 0 && (int) data.instance(i).classValue() == 1) {
					cost += mCost[0];
					temp1To0Num++;
				} else if (labels[i] == 1 && (int) data.instance(i).classValue() == 0) {
					cost += mCost[1];
					temp0To1Num++;
				} // Of else if
			} // of else
		} // of for
		System.out.println("(totalCost) The teach number is: " + tempTeachNum);
		System.out.println("(totalCost) The 1 to 0 number is: " + temp1To0Num);
		System.out.println("(totalCost) The 0 to 1 number is: " + temp0To1Num);
		return cost;
	}// of totalCost

	/**
	 ***********************************
	 * Test function. Test the totalCost.
	 ************************************
	 */
	public static void testTotalCost() {
		// Step 1. Prepare
		double[] tempMisclassificationCost = { 2, 4 };
		double tempTeachCost = 1;
		RJ_CAKU003 caku = new RJ_CAKU003("src/data/rj4instancetest.arff", tempMisclassificationCost, tempTeachCost);

		// Step 2. Set parameters.
		caku.instanceStatus[0] = 0;
		caku.instanceStatus[1] = 0;
		caku.instanceStatus[2] = 1;
		caku.instanceStatus[3] = 0;

		caku.labels[0] = 0;
		caku.labels[1] = 1;
		caku.labels[2] = 1;
		caku.labels[3] = 1;

		int[] tempActualLabels = new int[caku.data.numInstances()];
		for (int i = 0; i < caku.data.numInstances(); i++) {
			tempActualLabels[i] = (int) caku.data.instance(i).classValue();
		} // Of for i

		double tempTotalCost = caku.totalCost();
		System.out.println("(testTotalCost)The instance status is: " + Arrays.toString(caku.instanceStatus));
		System.out.println("(testTotalCost)The obtained labels are: " + Arrays.toString(caku.labels));
		System.out.println("(testTotalCost)The actual labels are: " + Arrays.toString(tempActualLabels));
		System.out.println("(testTotalCost)The total cost is: " + tempTotalCost);
	}// Of testTotalCost

	/**
	 *****************************
	 * Compute the distance between an object and an array
	 *****************************
	 */
	public double distance(int paraIndex, double[] paraArray) {
		double tempResultDistance = 0;
		for (int i = 0; i < paraArray.length; i++) {
			tempResultDistance += Math.abs(data.instance(paraIndex).value(i) - paraArray[i]);
		} // Of for i
		return tempResultDistance;
	}// Of distance

	/**
	 *****************************
	 * Is the given matrices equal? Judge the center and the new center is equal?
	 *****************************
	 */
	public static boolean doubleMatricesEqual(double[][] paraMatrix1, double[][] paraMatrix2) {
		for (int i = 0; i < paraMatrix1.length; i++) { // the number of line
			// the number of elements in a line
			for (int j = 0; j < paraMatrix1[0].length; j++) {
				// the precision is 10^-6
				if (Math.abs(paraMatrix1[i][j] - paraMatrix2[i][j]) > 1e-6) {
					return false;
				} // Of if
			} // Of for j
		} // Of for i
		return true;
	}// Of doubleMatricesEqual

	/**
	 ********************* 
	 * Compute the expect number of positive instances.
	 * 
	 * @param R the number of positive instances checked.
	 * @param B the number of negative instances checked.
	 * @param N the total number of instances.
	 * @return the expect number of positive instances.
	 * 
	 *         CADU equation 4
	 ********************* 
	 */
	public static double expectPosNum(int R, int B, int N) {
		BigDecimal numerator = new BigDecimal("0");
		BigDecimal denominator = new BigDecimal("0");
		for (int i = R; i <= N - B; i++) {
			BigDecimal a = A(R, i).multiply(A(B, N - i));
			numerator = numerator.add(a.multiply(new BigDecimal("" + i)));
			denominator = denominator.add(a);
			// System.out.println("(expectPosNum)a: " + a + ", numerator: " + numerator +
			// ", denominator: " + denominator );
		} // Of for i
		return numerator.divide(denominator, 4, BigDecimal.ROUND_HALF_EVEN).doubleValue();
	}// Of expectPosNum

	/**
	 ********************* 
	 * Compute arrangement of A^m_n where m <= B
	 * 
	 * compute the permutation
	 ********************* 
	 */
	public static BigDecimal A(int m, int n) {
		if (m > n) {
			return new BigDecimal("0");
		} // Of if
		BigDecimal re = new BigDecimal("1");
		for (int i = n - m + 1; i <= n; i++) {
			re = re.multiply(new BigDecimal(i));
		} // Of if
		return re;
	}// Of A

	/**
	 ************************************
	 * The method of integer-array to string
	 * 
	 * @param paraArray the target array
	 * @return the ideal string
	 ************************************
	 */
	public static String intArrayToString(int[] paraArray) {
		String tempString = "";
		for (int i = 0; i < paraArray.length - 1; i++) {
			tempString += paraArray[i] + ",";
		} // Of for i
		tempString += paraArray[paraArray.length - 1];
		return tempString;
	}// Of intArrayToString

	/**
	 ***********************************
	 * The testFunction: from index to instance
	 * 
	 * @param paraIndex The given index
	 * @return The instance from index
	 ************************************
	 */
	public double[][] indexToInstance(int[] paraIndex) {
		// System.out.println("(indexToInstance)The data attributes number is : " +
		// data.numAttributes());
		// System.out.println("(indexToInstance)The paraIndex is: " +
		// intArrayToString(paraIndex));
		double[][] tempInstances = new double[paraIndex.length][data.numAttributes() - 1];
		for (int i = 0; i < paraIndex.length; i++) {
			// System.out.println("(indexToInstance)The loop i :" + i);
			for (int j = 0; j < data.numAttributes() - 1; j++) {
				// System.out.println(data.instance(i).value(j));
				// System.out.println("(indexToInstance) The loop j :" + j);
				tempInstances[i][j] = data.instance(paraIndex[i]).value(j);
			} // of for j
		} // of for i
		return tempInstances;
	}// of indexToInstance

	/**
	 ***********************************
	 * Test function. Test the indexToInstance.
	 ************************************
	 */
	public static void testIndexToInstance() {

		// Step 1. Prepare
		double[] tempMisclassificationCost = { 2, 4 };
		double tempTeachCost = 1;
		RJ_CAKU003 caku = new RJ_CAKU003("data/rj4instancetest.arff", tempMisclassificationCost, tempTeachCost);
		int[] tempIndex = new int[4];
		for (int i = 0; i < tempIndex.length; i++) {
			tempIndex[i] = i;
		} // of for i
		System.out.println("(indexToInstance)The data in the test : " + caku.data);
		// Step 2. Test
		double[][] tempInstances = new double[tempIndex.length][caku.data.numAttributes() - 1];
		for (int i = 0; i < tempIndex.length; i++) {
			for (int j = 0; j < caku.data.numAttributes() - 1; j++) {
				// System.out.println(data.instance(i).value(j));
				tempInstances[i][j] = caku.data.instance(tempIndex[i]).value(j);
			} // of for j
		} // of for i

		System.out.println("(indexToInstance)The index : " + intArrayToString(tempIndex) + " to instance is "
				+ Arrays.deepToString(tempInstances));
	}// of testIndexToInstance

	/**
	 ***************
	 * Test Cluster. 2-Means Cluster split
	 * 
	 * @param paraBlock The to be processed array
	 * @param paraK     The K value of K-Means
	 ***************
	 */
	public int[] clusterTest(int[] paraBlock, double[][] paraCenters) {
		// Step 1. Initialize
		double tempLeastDistance;
		int paraK = 2;
		int tempBlockSize = paraBlock.length;
		int[] tempCluster = new int[tempBlockSize];
		double[][] tempNewCenters = paraCenters;
		double[][] tempCenters = new double[paraK][data.numAttributes() - 1];
		// Step 2. Cluster and compute new centers.
		int count = 0;
		while (!doubleMatricesEqual(tempCenters, tempNewCenters)) {
			// while (!Arrays.deepEquals(tempCenters, tempNewCenters)) {
			tempCenters = tempNewCenters;
			// Cluster
			for (int i = 0; i < tempBlockSize; i++) {
				tempLeastDistance = Double.MAX_VALUE;
				for (int j = 0; j < paraK; j++) {
					double tempDistance = distance(paraBlock[i], tempCenters[j]);
					System.out.println("(clusterTest)The " + i + " distance to instance[" + j + "] is " + tempDistance);
					if (tempDistance < tempLeastDistance) {
						System.out.println("(clusterTest)The instance[" + i + "] is blong to " + j);
						tempCluster[i] = j;
						if (j == 1) {
							count++;
						}
						System.out.println(
								"(clusterTest)The " + i + " distance to instance[" + j + "] is " + tempDistance);
						System.out.println("(clusterTest)The number of instnace blong to 1 is " + count);
						tempLeastDistance = tempDistance;
					} // Of if
				} // Of for j
			} // Of for i
			System.out.println("(clusterTest)Current cluster: " + Arrays.toString(tempCluster));
			// Compute new centers count the number of instances in different
			// class
			int[] tempCounters = new int[paraK];
			for (int i = 0; i < tempCounters.length; i++) {
				tempCounters[i] = 0;
			} // Of for i
				// 1. sum all in one kind
				// tempNewCenters = new double[paraK][data.numAttributes() - 1];
				// //why define tempNewCenter twice
			for (int i = 0; i < tempBlockSize; i++) {
				tempCounters[tempCluster[i]]++; // nice expect the center
				for (int j = 0; j < data.numAttributes() - 1; j++) {
					// include the center
					tempNewCenters[tempCluster[i]][j] += data.instance(paraBlock[i]).value(j); // include the center
				} // Of for j
			} // Of for i
			System.out.println("(clusterTest)............tempNewCenters is " + Arrays.deepToString(tempNewCenters));
			System.out.println("(clusterTest)            tempCounters " + Arrays.toString(tempCounters));
			// 2. Average Means conclude the new centers
			for (int i = 0; i < paraK; i++) {
				for (int j = 0; j < data.numAttributes() - 1; j++) {
					tempNewCenters[i][j] /= tempCounters[i];
				} // Of for j
			} // Of for i

			currentCenters = tempNewCenters;
			System.out.println("(clusterTest)----The currentCenters are" + Arrays.deepToString(currentCenters));
			System.out.println("(clusterTest)-----The centers are: " + Arrays.deepToString(tempCenters));
			System.out.println("(clusterTest)-----The new centers are: " + Arrays.deepToString(tempNewCenters));
		} // Of while
		return tempCluster;
	}// Of cluster
	
	/**
	 ***********************************
	 * The class value test
	 ************************************
	 */
	public static void testClassValue() {
		double[] tempMisclassificationCost = { 2, 4 };
		double tempTeachCost = 1;
		String dataTrace = "data/easyiris.arff";
		RJ_CAKU003 caku = new RJ_CAKU003(dataTrace, tempMisclassificationCost, tempTeachCost);
		for (int i = 0; i < caku.data.numInstances(); i++) {
			System.out.println("The " + i + " instance classValue is: " + caku.data.instance(i).classValue() );
		}//of for i
	}// of testClassValue
	
}// of RJ_CAKU003
