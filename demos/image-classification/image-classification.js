import {imagenetClasses} from 'imagenet.js';

/**
 * Find top k imagenet classes
 */
export function imagenetClassesTopK(classProbabilities, k = 5) {
    const probs = Array.isArray(classProbabilities) ? classProbabilities.slice() : classProbabilities;
  
    const sorted = probs
      .map((prob, index) => [prob, index])
      .sort((a, b) => b[0] - a[0]);
  
    const topK = sorted.slice(0, k).map(probIndex => {
      const iClass = imagenetClasses[probIndex[1]];
      return {
        id: iClass[0],
        index: parseInt(probIndex[1], 10),
        name: iClass[1].replace(/_/g, ' '),
        probability: probIndex[0]
      };
    });
    return topK;
  }