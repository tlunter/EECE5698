function [labels, cparams, sse] = kmeans(data, k, r)
first = true;

for curr_r = 1:r
    [new_labels, new_cparams, new_sse] = kmeans_once(data, k);
    if (first || new_sse(end) < sse(end))
        first = false;
        labels = new_labels;
        cparams = new_cparams;
        sse = new_sse;
    end
end

end

function [labels, cparams, sse] = kmeans_once(data,k)

cparams = initialize(data, k);
labels = zeros(k);
converged = false;
sse = [];
i = 0;

while (~converged)
    i = i + 1;
    labels = classify(data, k, cparams);
    new_cparams = recompute(data, k, labels);
    converged = check_converged(cparams, new_cparams, k);
    cparams = new_cparams;
    sse(i) = compute_sse(data, labels, cparams);
end

end

function [inits] = initialize(data,k)

[n, f] = size(data);

inits = [];

for curr_k = 1:k
    safe = false;
    while ~safe
        i = randi(n);
        mu = data(i,:);
        safe = true;
        
        for prev_k = 1:curr_k-1
            if (mu == inits(prev_k).mu)
                safe = false;
                break;
            end
        end
    end
    
    s = struct('mu', mu);
    inits = [inits; s];
end

end

function [labels] = classify(data, k, cparams)

[rows, features] = size(data);

dists = [];

for curr_x = 1:rows
    
    dist = [];
    for curr_k = 1:k
        dist_sqr = sum(power((data(curr_x, :) - cparams(curr_k).mu), 2));
        dist = [dist dist_sqr];
    end
    dists = [dists; dist];
end

labels = [];

for curr_dists = 1:rows
    dist = dists(curr_dists, :);
    
    min_dist = dist(1);
    min_dist_label = 1;
        
    for curr_k = 2:k
        if (min_dist > dist(curr_k))
            min_dist = dist(curr_k);
            min_dist_label = curr_k;
        end
    end
    labels = [labels; min_dist_label];
end

end

function [params] = recompute(data, k, labels)

[n, ~] = size(data);

params = [];

for curr_k = 1:k
    k_data = [];
    for curr_x = 1:n
        if (labels(curr_x) == curr_k)
            k_data = [k_data; data(curr_x, :)];
        end
    end
    mu = mean(k_data, 1);
    if (size(mu, 2) == 0)
        mu = 0.0;
    end
    s = struct('mu', mu);
    params = [params; s];
end

end

function converged = check_converged(old_params, new_params, k)

converged = true;

for curr_k = 1:k
    old_mu = old_params(curr_k).mu;
    new_mu = new_params(curr_k).mu;
    
    if (any(abs(old_mu - new_mu) > 0.000001))
        converged = false;
        break;
    end
end

end

function [sse] = compute_sse(data, labels, params)

[n, ~] = size(data);
sse = 0;

for curr_x = 1:n
    sse = sse + sum(power((data(curr_x, :) - params(labels(curr_x)).mu), 2));
end
end