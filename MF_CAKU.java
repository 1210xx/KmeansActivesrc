package classification;

import java.io.FileReader;
import java.lang.reflect.Array;
import java.math.BigDecimal;
import java.util.Arrays;
import java.util.Random;

import javax.xml.crypto.Data;

import weka.core.Instances;

/**
 * This is the CAKU algorithm implemented by Fan Min.
 * 
 * @author minfanphd
 *
 */
public class MF_CAKU {

	/**
	 * The data, including both conditional and decision attributes. The
	 * decision is binary, that is, the class label takes two values.
	 */
	Instances data;

	/**
	 * The misclassification costs setting. misclassificationCosts[0] for
	 * classifying positive to negative (0 to 1), and misclassificationCosts[1]
	 * for the reverse.
	 */
	double[] misclassificationCosts;

	/**
	 * The teach cost setting. It should be smaller than any element of
	 * misclassificationCosts.
	 */
	double teacherCost;

	/**
	 * The obtained/predicted labels. To be compared with true labels.
	 */
	int[] labels;

	/**
	 * The instance status. 0 for untreated, 1 for queried, and 2 for
	 * classified.
	 */
	int[] instanceStatus;

	/**
	 * The instance status. 0 for untreated.
	 */
	public static final int UNTREATED = 0;

	/**
	 * The instance status. 1 for queried.
	 */
	public static final int QUERIED = 1;

	/**
	 * The instance status. 2 for classified.
	 */
	public static final int CLASSIFIED = 2;

	/**
	 * The maximal number of instances that can be queried in one block.
	 */
	public static final int MAXIMAL_QUERY_ONE_BLOCK = 100;

	/**
	 * the random function
	 */
	Random random;

	/**
	 * the center of cluster
	 */
	// double[][] currentCenters;

	/**
	 *********************
	 * The constructor.
	 *********************
	 */
	public MF_CAKU(String paraFilename, double[] paraMisclassificationCosts, double paraTeacherCost) {

		// Read the arff file.
		data = null;
		try {
			FileReader fileReader = new FileReader(paraFilename);
			data = new Instances(fileReader);
			fileReader.close();
			data.setClassIndex(data.numAttributes() - 1);
		} catch (Exception ee) {
			System.out.println("Cannot read the file: " + paraFilename + "\r\n" + ee);
			System.exit(0);
		} // Of try

		// Accept parameters
		misclassificationCosts = paraMisclassificationCosts;
		teacherCost = paraTeacherCost;

		// Initialize other member variables.
		instanceStatus = new int[data.numInstances()];
		labels = new int[data.numInstances()];
		Arrays.fill(labels, -1);
		random = new Random();

		// Take a look.
		//System.out.println(data);
	}// Of the constructor

	/**
	 *************************************
	 * Compute the total cost.
	 *************************************
	 */
	public double computeTotalCost() {
		double tempTotalCost = 0;
		int tempQueried = 0;
		int tempMisclassified = 0;

		for (int i = 0; i < instanceStatus.length; i++) {
			if (instanceStatus[i] == UNTREATED) {
				System.out.println("Algorithm error! Instance " + i + " has not been treated yet.");
				System.exit(0);
			} else if (instanceStatus[i] == QUERIED) {
				tempQueried++;
				tempTotalCost += teacherCost;
			} else if (instanceStatus[i] == CLASSIFIED) {
				// Is the classification correct?
				if ((labels[i] == 0) && (int) data.instance(i).classValue() == 1) {
					tempTotalCost += misclassificationCosts[0];
					tempMisclassified++;
				} else if ((labels[i] == 1) && (int) data.instance(i).classValue() == 0) {
					tempTotalCost += misclassificationCosts[1];
					tempMisclassified++;
				} // Of if
			} else {
				System.out.println("System error! Instance " + i + " has an unrecognized status: " + instanceStatus[i]);
				System.exit(0);
			} // Of if
		} // Of for i

		System.out.println("Queried: " + tempQueried + ", misclassified: " + tempMisclassified);

		return tempTotalCost;
	}// Of computeTotalCost

	/**
	 *************************************
	 * Construct the original block with all instances.
	 *************************************
	 */
	int[] constructOriginalBlock() {
		int[] tempOriginalBlock = new int[data.numInstances()];
		for (int i = 0; i < tempOriginalBlock.length; i++) {
			tempOriginalBlock[i] = i;
		} // Of for i
		return tempOriginalBlock;
	}// Of constructOriginalBlock

	/**
	 *************************************
	 * Compute the Euclidean distance between an instance and a point.
	 * 
	 * @param paraInstanceIndex
	 *            The index of the instance.
	 * @param paraPoint
	 *            The given point, maybe not a concrete instance.
	 *************************************
	 */
	double euclideanDistance(int paraInstanceIndex, double[] paraPoint) {
		double tempDistance = 0;
		double tempDifference;

		for (int i = 0; i < paraPoint.length; i++) {
			tempDifference = data.instance(paraInstanceIndex).value(i) - paraPoint[i];
			tempDistance += tempDifference * tempDifference;
		} // Of for i

		return Math.sqrt(tempDistance);
	}// Of euclideanDistance

	/**
	 *************************************
	 * Compute the Euclidean distance between two instances
	 * 
	 * @param paraFirstIndex
	 *            The index of the first instance.
	 * @param paraSecondIndex
	 *            The index of the second instance.
	 *************************************
	 */
	double euclideanDistance(int paraFirstIndex, int paraSecondIndex) {
		double tempDistance = 0;
		double tempDifference;

		for (int i = 0; i < data.numAttributes() - 1; i++) {
			tempDifference = data.instance(paraFirstIndex).value(i) - data.instance(paraSecondIndex).value(i);
			tempDistance += tempDifference * tempDifference;
		} // Of for i

		return Math.sqrt(tempDistance);
	}// Of euclideanDistance

	/**
	 *************************************
	 * Split the block in 2 using kMeans
	 * 
	 * @param paraBlock
	 *            The given block.
	 * @param paraInitalCenterIndices
	 *            The indices of the given 2 centers.
	 * @return 2 blocks after splitting.
	 *************************************
	 */
	int[][] twoMeansClustering(int[] paraBlock, int[] paraInitalCenterIndices) {
		// Allocate enough space to avoid reallocating.
		int[][] tempBlocks = new int[2][paraBlock.length];
		// The size control the valid number of elements.
		int[] tempSizes = new int[2];

		int[][] tempNewBlocks = null;

		double[][] tempCenters = new double[2][data.numAttributes() - 1];
		double[][] tempNewCenters = new double[2][data.numAttributes() - 1];
		for (int i = 0; i < 2; i++) {
			for (int j = 0; j < data.numAttributes() - 1; j++) {
				tempNewCenters[i][j] = data.instance(paraInitalCenterIndices[i]).value(j);
			} // Of for j
		} // Of for i

		boolean tempConverged = false;
		// Iterate until the centers do not change.
		while (!tempConverged) {
			Arrays.fill(tempSizes, 0);

			// Step 1. Move instances to blocks.
			for (int i = 0; i < paraBlock.length; i++) {
				if (euclideanDistance(paraBlock[i], tempNewCenters[0]) < euclideanDistance(paraBlock[i],
						tempNewCenters[1])) {
					tempBlocks[0][tempSizes[0]] = paraBlock[i];
					tempSizes[0]++;
				} else {
					tempBlocks[1][tempSizes[1]] = paraBlock[i];
					tempSizes[1]++;
				} // Of if
			} // Of for i
				// System.out.println("tempSizes: " +
				// Arrays.toString(tempSizes));

			// Step 2. Compute new centers.
			tempNewCenters = new double[2][data.numAttributes() - 1];
			// Two centers.
			for (int i = 0; i < 2; i++) {
				// Sum
				for (int j = 0; j < tempSizes[i]; j++) {
					for (int k = 0; k < data.numAttributes() - 1; k++) {
						tempNewCenters[i][k] += data.instance(tempBlocks[i][j]).value(k);
					} // Of for k
				} // Of for j

				// Average
				for (int k = 0; k < data.numAttributes() - 1; k++) {
					tempNewCenters[i][k] /= tempSizes[i];
				} // Of for k
			} // Of for i

			// Step 3. Judge again and update centers.
			// System.out.println("TempCenters:"
			// + doubleMatrixToString(tempCenters, 5, ','));
			// System.out.println("TempNewCenters:"
			// + doubleMatrixToString(tempNewCenters, 5, ','));
			tempConverged = doubleMatricesEqual(tempCenters, tempNewCenters);
			if (tempConverged) {
				break;
			} else {
				tempCenters = tempNewCenters;
			} // Of if
		} // Of while

		// Now compress for return.
		tempNewBlocks = new int[2][];
		for (int i = 0; i < 2; i++) {
			tempNewBlocks[i] = new int[tempSizes[i]];
			for (int j = 0; j < tempSizes[i]; j++) {
				tempNewBlocks[i][j] = tempBlocks[i][j];
			} // Of for j
		} // Of for i
		System.out.println("The cluster array is " + Arrays.deepToString(tempNewBlocks));
		return tempNewBlocks;
	}// Of twoMeansClustering

	/**
	 *************************************
	 * Are the given double matrices equal?
	 * 
	 * @param paraFirstMatrix
	 *            The first matrix.
	 * @param paraSecondMatrix
	 *            The second matrix.
	 * @return True if equal.
	 *************************************
	 */
	public static boolean doubleMatricesEqual(double[][] paraFirstMatrix, double[][] paraSecondMatrix) {
		// Same number of rows?
		if (paraFirstMatrix.length != paraSecondMatrix.length) {
			return false;
		} // Of if

		// Same number of columns?
		if (paraFirstMatrix[0].length != paraSecondMatrix[0].length) {
			return false;
		} // Of if

		// Now check all elements.
		for (int i = 0; i < paraFirstMatrix.length; i++) {
			for (int j = 0; j < paraFirstMatrix[0].length; j++) {
				if (Math.abs(paraFirstMatrix[i][j] - paraSecondMatrix[i][j]) > 1e-6) {
					return false;
				} // Of if
			} // Of for j
		} // Of for i

		return true;
	}// Of doubleMatricesEqual

	/**
	 *************************************
	 * Test the method.
	 *************************************
	 */
	public static void testDoubleMatricesEqual() {
		double[][] tempFirstMatrix = { { 0.300001, 0.4 }, { 0.1, 0.2 } };
		double[][] tempSecondMatrix = { { -4, -3 }, { -2, -1 } };

		boolean tempEqual = false;
		try {
			tempEqual = doubleMatricesEqual(tempFirstMatrix, tempSecondMatrix);
		} catch (Exception ee) {
			tempEqual = false;
		} // Of try

		System.out.println("Are the matrices equal?" + tempEqual);
	}// Of testDoubleMatricesEqual

	/**
	 ************************* 
	 * Find the next farthest instance of the block.
	 * 
	 * @param paraBlock
	 *            The given block.
	 * @param paraLabeledInstances
	 *            Labeled instances in the current block. Some elements at the
	 *            tail are essentially empty.
	 * @param paraActualQueried
	 *            The number of actual queried instances in the current block.
	 ************************* 
	 */
	public int findFarthest(int[] paraBlock, int[] paraLabeledInstances, int paraActualQueried) {
		int resultFarthest = -1;
		double tempMaxDistanceSum = -1;
		for (int i = 0; i < paraBlock.length; i++) {
			double tempDistanceSum = 0;
			for (int j = 0; j < paraActualQueried; j++) {
				if (paraBlock[i] == paraLabeledInstances[j]) {
					tempDistanceSum = -1;
					break;
				} // Of if

				tempDistanceSum += euclideanDistance(paraBlock[i], paraLabeledInstances[j]);
			} // Of for j

			// Update
			if (tempDistanceSum > tempMaxDistanceSum + 1e-6) {
				resultFarthest = paraBlock[i];
				tempMaxDistanceSum = tempDistanceSum;
			} // Of if
		} // Of for i

		return resultFarthest;
	}// Of findFarthest

	/**
	 ***********************************
	 * The main progress. Query a number of labels. If the current queried label
	 * is different from the first one, split the block. Otherwise classify all
	 * remaining instances in the block.
	 * 
	 * @param paraBlock
	 *            the array to be processed
	 * @param paraInitialPoint
	 *            the begin instances index Unfinished yet.
	 ************************************
	 */
	public void querySplitClassify(int[] paraBlock, int paraInitialPoint) {
		System.out.println("Trying to handle a block with " + paraBlock.length + " instances with initial point "
				+ paraInitialPoint + "\rAnd  the block is :" + Arrays.toString(paraBlock));
		// The first label. All other queried labels should be equal to it.
		int tempFirstLabel = -1;
		boolean tempFirstLabelKnown = false;
		// Is this block pure?
		boolean tempPure = true;

		// Labeled instances.
		int[] tempQueriedInstances = new int[MAXIMAL_QUERY_ONE_BLOCK];
		int tempNumQueried = 0;

		// Used to split.
		int[] tempBrothers = new int[2];
		Arrays.fill(tempBrothers, -1);
		// Two sub-blocks.
		int[][] tempSubBlocks;

		// Step 1. Is the initial point in this block.
		// Maybe not due to the 2-means algorithm.
		// boolean tempInitialPointInBlock = false;
		for (int i = 0; i < paraBlock.length; i++) {
			if (paraBlock[i] == paraInitialPoint) {
				// tempInitialPointInBlock = true;

				// The initial point should be labeled.
				if (instanceStatus[paraInitialPoint] != QUERIED) {
					instanceStatus[paraInitialPoint] = QUERIED;
					labels[paraInitialPoint] = (int) data.instance(paraInitialPoint).classValue();
				} // Of if

				tempFirstLabel = labels[paraInitialPoint];
				tempBrothers[0] = paraInitialPoint;
				tempFirstLabelKnown = true;

				tempQueriedInstances[tempNumQueried] = paraInitialPoint;
				tempNumQueried++;
				break;
			} // Of if
		} // Of for i

		// Step 2. Count the queried instance
		for (int i = 0; i < paraBlock.length; i++) {
			if (instanceStatus[paraBlock[i]] == QUERIED) {
				// It has been handled already.
				if (paraBlock[i] == paraInitialPoint) {
					continue;
				} // Of if

				tempQueriedInstances[tempNumQueried] = paraInitialPoint;
				tempNumQueried++;

				// Use the first known label if unspecified by the parameter paraInitialPoint.
				if (!tempFirstLabelKnown) {
					tempFirstLabel = labels[paraBlock[i]];
					tempBrothers[0] = paraBlock[i];
					tempFirstLabelKnown = true;
				} // Of if

				// Is it pure?
				if (labels[paraBlock[i]] != tempFirstLabel) {
					tempBrothers[1] = paraBlock[i];
					tempPure = false;
					break;
				} // Of if
			} // Of if
		} // Of for i
		System.out.println("" + tempNumQueried + " instances have been queried in this block.");
		System.out.println("Are they pure? " + tempPure);

		// Step 3. Split if impure.
		if (!tempPure) {
			System.out.println("The brothes are: " + Arrays.toString(tempBrothers));
			tempSubBlocks = twoMeansClustering(paraBlock, tempBrothers);

			System.out.println("Split in two: " + tempSubBlocks[0].length + " and " + tempSubBlocks[1].length);
			querySplitClassify(tempSubBlocks[0], tempBrothers[0]);
			querySplitClassify(tempSubBlocks[1], tempBrothers[1]);

			return;
		} // Of if

		// Step 4. No label has been queried, choose the first to query.
		if (tempNumQueried == 0) {
			labels[paraBlock[0]] = (int) data.instance(paraBlock[0]).classValue();
			instanceStatus[paraBlock[0]] = QUERIED;
			tempQueriedInstances[tempNumQueried] = paraBlock[0];
			tempFirstLabel = labels[paraBlock[0]];
			tempBrothers[0] = paraBlock[0];
			tempNumQueried++;
		} // Of if

		// Step 5. How many labels should be queried?
		int[] tempLookups = lookup(paraBlock.length);
		//int tempNeededQueries = (int)Math.sqrt(paraBlock.length)
		int tempNeededQueries = tempLookups[tempFirstLabel] - tempNumQueried;
		System.out.println("------Need to buy :" + tempNeededQueries);

		// Step 6. Query enough labels.
		int tempNextToQuery;
		for (int i = 0; i < tempNeededQueries; i++) {
			// Find the next to query.
			tempNextToQuery = findFarthest(paraBlock, tempQueriedInstances, tempNumQueried);

			// Query
			tempQueriedInstances[tempNumQueried] = tempNextToQuery;
			instanceStatus[tempNextToQuery] = QUERIED;
			labels[tempNextToQuery] = (int) data.instance(tempNextToQuery).classValue();
			tempNumQueried++;

			// Split if impure
			System.out.print(", " + tempNextToQuery + "(" + labels[tempNextToQuery] + ")");
			if (labels[tempNextToQuery] != tempFirstLabel) {
				System.out.println("#" + tempNextToQuery + " has a different label: " + labels[tempNextToQuery]);
				tempBrothers[1] = tempNextToQuery;
				tempSubBlocks = twoMeansClustering(paraBlock, tempBrothers);

				System.out.println("Split in two: " + tempSubBlocks[0].length + " and " + tempSubBlocks[1].length);
				System.out.println("#2 The brothes are: " + Arrays.toString(tempBrothers));

				querySplitClassify(tempSubBlocks[0], tempBrothers[0]);
				querySplitClassify(tempSubBlocks[1], tempBrothers[1]);

				return;
			} // Of if
		} // Of for i

		// Step 7. Classify others
		System.out.println("This block is " + tempPure + ". Classify all others instances as " + tempFirstLabel);
		for (int i = 0; i < paraBlock.length; i++) {
			if (instanceStatus[paraBlock[i]] == UNTREATED) {
				instanceStatus[paraBlock[i]] = CLASSIFIED;
				labels[paraBlock[i]] = tempFirstLabel;
			} // Of if
		} // Of for i
	}// Of querySplitClassify

	/**
	 *************************** 
	 * Convert an integer matrix into a string. Integers are separated by
	 * separators. Author Xiangju Li.
	 * 
	 * @param paraMatrix
	 *            The given matrix.
	 * @param paraSeparator
	 *            The separator of data, blank and commas are most commonly uses
	 *            ones.
	 * @return The constructed String.
	 *************************** 
	 */
	public static String intMatrixToString(int[][] paraMatrix, char paraSeparator) {
		String resultString = "[]";
		if ((paraMatrix == null) || (paraMatrix.length < 1))
			return resultString;
		resultString = "[";
		for (int i = 0; i < paraMatrix.length; i++) {
			resultString += "[";
			for (int j = 0; j < paraMatrix[i].length - 1; j++) {
				resultString += "" + paraMatrix[i][j] + paraSeparator + " ";
			} // Of for j
			resultString += paraMatrix[i][paraMatrix.length - 1];
			if (i == paraMatrix.length - 1) {
				resultString += "]]";
			} else {
				resultString += "]\r\n";
			} // Of if
		} // Of for i

		return resultString;
	}// Of intMatrixToString

	/**
	 *************************** 
	 * Convert a double matrix into a string. Integers are separated by
	 * separators.
	 * 
	 * @param paraMatrix
	 *            The given matrix.
	 * @param paraDigits
	 *            How many digits after the dot.
	 * @param paraSeparator
	 *            The separator of data, blank and commas are most commonly uses
	 *            ones.
	 * @return The constructed String.
	 *************************** 
	 */
	public static String doubleMatrixToString(double[][] paraMatrix, int paraDigits, char paraSeparator) {
		String resultString = "[]";
		if ((paraMatrix == null) || (paraMatrix.length < 1))
			return resultString;
		resultString = "[";
		double tempValue = Math.pow(10, paraDigits);

		for (int i = 0; i < paraMatrix.length; i++) {
			resultString += "[";
			for (int j = 0; j < paraMatrix[i].length - 1; j++) {
				resultString += "" + ((int) (paraMatrix[i][j] * tempValue) + 0.0) / tempValue + paraSeparator + " ";
			} // Of for j

			resultString += ((int) (paraMatrix[i][paraMatrix.length - 1] * 10000) + 0.0) / 10000;
			if (i == paraMatrix.length - 1) {
				resultString += "]]";
			} else {
				resultString += "]\r\n";
			} // Of if
		} // Of for i

		return resultString;
	}// Of intMatrixToString

	/**
	 ********************* 
	 * Look up optimal R and B
	 * 
	 * @param paraBlockSize
	 *            the input block size
	 * @author Yan-Xue Wu
	 ********************* 
	 */
	private int[] lookup(int paraBlockSize) {
		// Linear Search
		// star[] :note the tmpMinCost instance in every query
		// starLoose: the final MinCost instance index
		// ra :the expect number of positive instances
		// ra[0] : compute the final starLoose as the compute the tmpCost
		// isFind : find the MinCost instance
		double[] tmpMinCost = new double[] { 0.5 * misclassificationCosts[0] * paraBlockSize,
				0.5 * misclassificationCosts[1] * paraBlockSize };
		// double[] tmpMinCost = new double[] {mCost[0] * paraBlockSize,
		// mCost[1] * paraBlockSize }; //pair with the starLoose computation so
		// use the 0.5
		int[] star = new int[2];
		int[] starLoose = new int[2];
		boolean[] isFind = new boolean[2];
		double[] ra = new double[paraBlockSize + 1];
		ra[0] = 0.5;
		double[] tmpCost = new double[2];
		for (int i = 1; i <= paraBlockSize; i++) {
			if (paraBlockSize >= 1000) {
				// the expect numbers of positive instances similarly the N
				// approach infinite
				ra[i] = (i + 1.0) / (i + 2.0);
			} else {
				// else use the CADU equation (4) to compute the ra
				ra[i] = expectPosNum(i, 0, paraBlockSize) / paraBlockSize;
			}
			for (int j = 0; j < 2; j++) {
				tmpCost[j] = (1 - ra[i]) * misclassificationCosts[j] * paraBlockSize + teacherCost * i;
				if (tmpCost[j] < tmpMinCost[j]) {
					tmpMinCost[j] = tmpCost[j];
					star[j] = i;
					if (i == paraBlockSize) {
						Arrays.fill(isFind, true);
					} // Of if
				} else {
					isFind[j] = true;
				} // Of if
			} // Of for j

			// System.out.println("star : " + intArrayToString(star));
			// System.out.println("cost :" + Arrays.toString(tmpCost));
			// System.out.println("QueriedInstance: " + i + ", cost: " +
			// tmpCost[0] + "\t");
			if (isFind[0] && isFind[1]) {
				Arrays.fill(isFind, false);
				for (int j = 0; j <= i; j++) {
					for (int k = 0; k < 2; k++) {
						tmpCost[k] = (1 - ra[j]) * misclassificationCosts[k] * paraBlockSize + teacherCost * j;
						if (tmpCost[k] <= tmpMinCost[k] && !isFind[k]) {
							isFind[k] = true;
							starLoose[k] = j; //Dangerous
						} // Of if
						if (isFind[0] && isFind[1]) {
							// System.out.println("------starLoose is : " +
							// intArrayToString(starLoose));
							return starLoose;
						} // Of if
					} // Of for k
				} // Of for j
			} // Of if
		} // Of for i
		return new int[] { 0, 0 };
		// throw new
		// RuntimeException("Error occured in lookup("+paraBlockSize+")");
	}// Of lookup

	/**
	 ********************* 
	 * Compute the expect number of positive instances.
	 * 
	 * @param R
	 *            the number of positive instances checked.
	 * @param B
	 *            the number of negative instances checked.
	 * @param N
	 *            the total number of instances.
	 * @return the expect number of positive instances.
	 * 
	 *         CADU equation 4
	 * @author Yan-Xue Wu
	 ********************* 
	 */
	public static double expectPosNum(int R, int B, int N) {
		BigDecimal numerator = new BigDecimal("0");
		BigDecimal denominator = new BigDecimal("0");
		for (int i = R; i <= N - B; i++) {
			BigDecimal a = A(R, i).multiply(A(B, N - i));
			numerator = numerator.add(a.multiply(new BigDecimal("" + i)));
			denominator = denominator.add(a);
			// System.out.println("a: " + a + ", numerator: " + numerator +
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
	 *************************************
	 * The main entrance.
	 * 
	 * @param args
	 *            Not used outside the command mode.
	 *************************************
	 */
	public static void main(String[] args) {
		// testDoubleMatricesEqual();

		// Set and construct the object.
		double[] tempMisclassificationCosts = { 2, 4 };
		double tempTeacherCost = 1;
		// MF_CAKU caku = new MF_CAKU("src/data/rj4instances.arff",
		MF_CAKU caku = new MF_CAKU("data/CAKU.arff",
				//MF_CAKU caku = new MF_CAKU("src/data/easyiris.arff",
				tempMisclassificationCosts, tempTeacherCost);

		/*
		 * System.out.println("Test clustering begins."); int[]
		 * tempCenterIndices = { 1, 60 }; int[][] tempBlocks =
		 * caku.twoMeansClustering( caku.constructOriginalBlock(),
		 * tempCenterIndices); System.out.println("The blocks are: " +
		 * intMatrixToString(tempBlocks, ','));
		 * System.out.println("Test clustering ends.");
		 */

		//Construct the whole block.
		int[] tempInitialBlock = new int[caku.data.numInstances()];
		for (int i = 0; i < tempInitialBlock.length; i++) {
			tempInitialBlock[i] = i;
		} // Of for i

		//Invoke the main method to learn.
		caku.querySplitClassify(tempInitialBlock, 0);

		// double tempTotalCost = caku.totalCost();
		double tempTotalCost = caku.computeTotalCost();
		System.out.println("The total instance number is : " + caku.data.numInstances());
		System.out.println("The averge cost is: " + tempTotalCost);
		System.out.println("Done.");
	}// Of main

}// Of class MF_CAKU
