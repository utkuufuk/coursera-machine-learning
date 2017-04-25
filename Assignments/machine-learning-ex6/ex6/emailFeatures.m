function x = emailFeatures(wordIndices)
%Takes in a word_indices vector and produces a feature vector from the word indices

    DICTIONARY_SIZE = 1899;
    x = zeros(1, DICTIONARY_SIZE);

    for w = 1:DICTIONARY_SIZE
       
        x(w) = sum(wordIndices == w) > 0;
    end
end
