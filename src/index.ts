import * as math from 'mathjs'
import csvToMatrix from 'csv-to-array-matrix'

// Use this instead of mathjs-util
const getDimensionSize = (matrix: any) => matrix[0].length

const getMeanAsRowVector = (matrix: any) => {
  const n = getDimensionSize(matrix)
  const vectors = Array(n)
    .fill(0)
    .map((_: any, i: number) => math.evaluate(`matrix[:, ${i + 1}]`, { matrix }))

  return vectors.reduce((result: any, vector: any) => result.concat(math.mean(vector)), [])
}

const getStdAsRowVector = (matrix: any) => {
  const n = getDimensionSize(matrix)
  const vectors = Array(n)
    .fill(0)
    .map((_: any, i: number) => math.evaluate(`matrix[:, ${i + 1}]`, { matrix }))
  return vectors.reduce((result: any, vector: any) => result.concat(math.std(vector)), [])
}

csvToMatrix('./src/data.csv', init)

function init(matrix: any) {
  // Part 0: Preparation
  console.log('Part 0: Preparation ...\n')

  let X = math.evaluate('matrix[:, 1:2]', {
    matrix
  })
  let y = math.evaluate('matrix[:, 3]', {
    matrix
  })

  let m = y.length

  // Part 1: Feature Normalization
  console.log('Part 1: Feature Normalization ...\n')

  let { XNorm, mu, sigma } = featureNormalize(X)

  console.log('X normalized: ', XNorm)
  console.log('\n')
  console.log('mean: ', mu)
  console.log('\n')
  console.log('std: ', sigma)
  console.log('\n')

  // Part 2: Gradient Descent
  console.log('Part 2: Gradient Descent ...\n')

  // Add Intercept Term
  XNorm = X = math.concat(math.ones([m, 1]).valueOf() as any, XNorm)

  const ALPHA = 0.01
  const ITERATIONS = 400

  let theta = [[0], [0], [0]]
  theta = gradientDescentMulti(XNorm, y, theta, ALPHA, ITERATIONS)

  console.log('theta: ', theta)
  console.log('\n')

  // Part 3: Predict Price of 1650 square meter and 3 bedroom house
  console.log('Part 3: Price Prediction ...\n')

  let normalizedHouseVector = [1, (1650 - mu[0]) / sigma[0], (3 - mu[1]) / sigma[1]]
  let price = math.evaluate('normalizedHouseVector * theta', {
    normalizedHouseVector,
    theta
  })

  console.log('Predicted price for a 1650 square meter and 3 bedroom house: ', price)
}

function featureNormalize(X: any) {
  const mu = getMeanAsRowVector(X)
  const sigma = getStdAsRowVector(X) // alternative: range
  console.log(mu)

  // n = features
  const n = X[0].length
  for (let i = 0; i < n; i++) {
    let featureVector = math.evaluate(`X[:, ${i + 1}]`, {
      X
    })

    let featureMeanVector = math.evaluate('featureVector - mu', {
      featureVector,
      mu: mu[i]
    })

    let normalizedVector = math.evaluate('featureMeanVector / sigma', {
      featureMeanVector,
      sigma: sigma[i]
    })

    math.evaluate(`X[:, ${i + 1}] = normalizedVector`, {
      X,
      normalizedVector
    })
  }

  return { XNorm: X, mu, sigma }
}

function gradientDescentMulti(X: number[][], y: number[], theta: any, ALPHA: number, ITERATIONS: number) {
  const m = y.length

  for (let i = 0; i < ITERATIONS; i++) {
    theta = math.evaluate(`theta - ALPHA / m * ((X * theta - y)' * X)'`, {
      theta,
      ALPHA,
      m,
      X,
      y
    })
  }

  return theta
}
