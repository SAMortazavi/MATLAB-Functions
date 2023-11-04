function value = sampen(signal, m, r, dist_type)

    % Error detection and defaults
    if nargin < 3, error('Not enough parameters.'); end
    if nargin < 4
        dist_type = 'chebychev';
        fprintf('[WARNING] Using default distance method: chebychev.\n');
    end
    if ~isvector(signal)
        error('The signal parameter must be a vector.');
    end
    if ~ischar(dist_type)
        error('Distance must be a string.');
    end
    if m > length(signal)
        error('Embedding dimension must be smaller than the signal length (m<N).');
    end
    
    % Useful parameters
    signal = signal(:)';
    N = length(signal);     % Signal length
    sigma = std(signal);    % Standard deviation
    
    % Create the matrix of matches
    matches = NaN(m+1,N);
    for i = 1:1:m+1
        matches(i,1:N+1-i) = signal(i:end);
    end
    matches = matches';

    % Check the matches for m
    d_m = pdist(matches(:,1:m), dist_type);
    if isempty(d_m)
        % If B = 0, SampEn is not defined: no regularity detected
        %   Note: Upper bound is returned
        value = Inf;
    else
        % Check the matches for m+1
        d_m1 = pdist(matches(:,1:m+1), dist_type);
        
        % Compute A and B
        %   Note: logical operations over NaN values are always 0
        B = sum(d_m  <= r*sigma);
        A = sum(d_m1 <= r*sigma);

        % Sample entropy value
        %   Note: norm. comes from [nchoosek(N-m+1,2)/nchoosek(N-m,2)]
        value = -log((A/B)*((N-m+1)/(N-m-1))); 
    end
    
    % If A=0 or B=0, SampEn would return an infinite value. However, the
    % lowest non-zero conditional probability that SampEn should
    % report is A/B = 2/[(N-m-1)(N-m)]
    if isinf(value)
        % Note: SampEn has the following limits:
        %       - Lower bound: 0
        %       - Upper bound: log(N-m)+log(N-m-1)-log(2)
        value = -log(2/((N-m-1)*(N-m)));
    end
end