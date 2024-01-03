function samples = generate_random_samples(mean, covariance_matrix, num_samples)
    % Ensure the mean vector is a column vector
    mean = mean(:);
    
    % Generate random samples
    samples = mvnrnd(mean, covariance_matrix, num_samples);
end

% usage example
% Define mean vector and covariance matrix
mean_vector = [1; 2];
covariance_matrix = [3, 0.5; 0.5, 1];

% Generate 5 random samples
num_samples = 5;
random_samples = generate_random_samples(mean_vector, covariance_matrix, num_samples);

disp('Generated Random Samples:');
disp(random_samples);

% -----------------------------------------------------

function discriminant = discriminant_function(x, mean, covariance, prior_probability)
    % Ensure input vectors are column vectors
    x = x(:);
    mean = mean(:);

    % Calculate Mahalanobis distance
    cov_inv = inv(covariance);
    mahalanobis_distance = sqrt((x - mean).' * cov_inv * (x - mean));

    % Calculate the discriminant function
    discriminant = -0.5 * mahalanobis_distance^2 - 0.5 * log(det(covariance)) + log(prior_probability);
end

% usage example
% Define mean vector, covariance matrix, and prior probability
mean_vector = [1; 2];
covariance_matrix = [3, 0.5; 0.5, 1];
prior_probability = 0.5;

% Define input vector
input_vector = [0.5; 1.5];

% Calculate the discriminant function value
discriminant_result = discriminant_function(input_vector, mean_vector, covariance_matrix, prior_probability);

disp(['Discriminant Function Value: ' num2str(discriminant_result)]);

% -----------------------------------------------------

% MATLAB function for Euclidean distance
function distance = euclidean_distance(point1, point2)
    % Ensure points are row vectors
    point1 = point1(:)';
    point2 = point2(:)';

    % Calculate Euclidean distance
    distance = sqrt(sum((point2 - point1).^2));
end

% usage example
point_a = [1, 2];
point_b = [4, 6];

distance_result = euclidean_distance(point_a, point_b);
disp(['Euclidean Distance: ' num2str(distance_result)]);

% -----------------------------------------------------

% MATLAB function for Mahalanobis distance
function distance = mahalanobis_distance(x, mean, covariance)
    % Ensure input vectors are column vectors
    x = x(:);
    mean = mean(:);

    % Calculate Mahalanobis distance
    cov_inv = inv(covariance);
    distance = sqrt((x - mean)' * cov_inv * (x - mean));
end

% usage example
mean_vector = [1; 2];
covariance_matrix = [3, 0.5; 0.5, 1];
input_vector = [0.5; 1.5];

mahalanobis_result = mahalanobis_distance(input_vector, mean_vector, covariance_matrix);
disp(['Mahalanobis Distance: ' num2str(mahalanobis_result)]);



