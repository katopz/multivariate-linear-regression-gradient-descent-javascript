"use strict";
var __importStar = (this && this.__importStar) || function (mod) {
    if (mod && mod.__esModule) return mod;
    var result = {};
    if (mod != null) for (var k in mod) if (Object.hasOwnProperty.call(mod, k)) result[k] = mod[k];
    result["default"] = mod;
    return result;
};
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const math = __importStar(require("mathjs"));
const csv_to_array_matrix_1 = __importDefault(require("csv-to-array-matrix"));
// Use this instead of mathjs-util
const getDimensionSize = (matrix) => matrix[0].length;
const getMeanAsRowVector = (matrix) => {
    const n = getDimensionSize(matrix);
    const vectors = Array(n)
        .fill(0)
        .map((_, i) => math.evaluate(`matrix[:, ${i + 1}]`, { matrix }));
    return vectors.reduce((result, vector) => result.concat(math.mean(vector)), []);
};
const getStdAsRowVector = (matrix) => {
    const n = getDimensionSize(matrix);
    const vectors = Array(n)
        .fill(0)
        .map((_, i) => math.evaluate(`matrix[:, ${i + 1}]`, { matrix }));
    return vectors.reduce((result, vector) => result.concat(math.std(vector)), []);
};
csv_to_array_matrix_1.default('./src/data.csv', init);
function init(matrix) {
    // Part 0: Preparation
    console.log('Part 0: Preparation ...\n');
    let X = math.evaluate('matrix[:, 1:2]', {
        matrix
    });
    let y = math.evaluate('matrix[:, 3]', {
        matrix
    });
    let m = y.length;
    // Part 1: Feature Normalization
    console.log('Part 1: Feature Normalization ...\n');
    let { XNorm, mu, sigma } = featureNormalize(X);
    console.log('X normalized: ', XNorm);
    console.log('\n');
    console.log('mean: ', mu);
    console.log('\n');
    console.log('std: ', sigma);
    console.log('\n');
    // Part 2: Gradient Descent
    console.log('Part 2: Gradient Descent ...\n');
    // Add Intercept Term
    XNorm = X = math.concat(math.ones([m, 1]).valueOf(), XNorm);
    const ALPHA = 0.01;
    const ITERATIONS = 400;
    let theta = [[0], [0], [0]];
    theta = gradientDescentMulti(XNorm, y, theta, ALPHA, ITERATIONS);
    console.log('theta: ', theta);
    console.log('\n');
    // Part 3: Predict Price of 1650 square meter and 3 bedroom house
    console.log('Part 3: Price Prediction ...\n');
    let normalizedHouseVector = [1, (1650 - mu[0]) / sigma[0], (3 - mu[1]) / sigma[1]];
    let price = math.evaluate('normalizedHouseVector * theta', {
        normalizedHouseVector,
        theta
    });
    console.log('Predicted price for a 1650 square meter and 3 bedroom house: ', price);
}
function featureNormalize(X) {
    const mu = getMeanAsRowVector(X);
    const sigma = getStdAsRowVector(X); // alternative: range
    console.log(mu);
    // n = features
    const n = X[0].length;
    for (let i = 0; i < n; i++) {
        let featureVector = math.evaluate(`X[:, ${i + 1}]`, {
            X
        });
        let featureMeanVector = math.evaluate('featureVector - mu', {
            featureVector,
            mu: mu[i]
        });
        let normalizedVector = math.evaluate('featureMeanVector / sigma', {
            featureMeanVector,
            sigma: sigma[i]
        });
        math.evaluate(`X[:, ${i + 1}] = normalizedVector`, {
            X,
            normalizedVector
        });
    }
    return { XNorm: X, mu, sigma };
}
function gradientDescentMulti(X, y, theta, ALPHA, ITERATIONS) {
    const m = y.length;
    for (let i = 0; i < ITERATIONS; i++) {
        theta = math.evaluate(`theta - ALPHA / m * ((X * theta - y)' * X)'`, {
            theta,
            ALPHA,
            m,
            X,
            y
        });
    }
    return theta;
}
//# sourceMappingURL=index.js.map