function [actual_label] = plot_data(data, labels, cparam)

kmeans = false;
if (size(labels, 2) == 1)
    kmeans = true;
end

rows = size(data, 1);

actual_label = zeros(rows, 1);

colors = 'crgybkm';
marker_colors = 'bkmrcgy';
markers = 'o+*.xs^';
cla;
hold on;
if (kmeans)
    for i = 1:rows
        index = labels(i);

        color = colors(index);
        marker = markers(index);

        plot(data(i, 1), data(i, 2), [color marker]);
    end
else
    for i = 1:rows
        [~, index] = max(labels(i, :));
        
        actual_label(i) = index;

        color = colors(index);
        marker = markers(index);

        plot(data(i, 1), data(i, 2), [color marker]);
    end
end

if (size(cparam) > 0)
    for i = 1:size(cparam, 1)
        color = marker_colors(i);
        marker = markers(i);
        plot(cparam(i).mu(1), cparam(i).mu(2), [color marker], 'MarkerSize', 12);
        if (~kmeans)
            % Calculate contours for the 2d normals at Mahalanobis dist = constant
            mhdist = 3;

            % Extract the relevant dimensions from the ith component matrix
            covar2d = [cparam(i).covar(1,1) cparam(i).covar(1,2); cparam(i).covar(2,1) cparam(i).covar(2,2)];

            % Use some results from standard geometry to figure out the ellipse
            % equations from the covariance matrix. Probably other ways to
            % do this, e.g., finding the principal component directions, etc.
            % See Fraleigh, p.431 for details on rotating the ellipse, etc
            icov = inv(covar2d);
            a = icov(1,1);
            c = icov(2,2);
            % we don't check if this is zero: which occasionally causes
            % problems when we divide by it later! needs to be fixed.
            b = icov(1,2)*2;

            theta = 0.5*acot( (a-c)/b);

            sc = sin(theta)*cos(theta);
            c2 = cos(theta)*cos(theta);
            s2 = sin(theta)*sin(theta);

            a1 = a*c2 + b*sc + c*s2;
            c1 = a*s2 - b*sc + c*c2;

            th= 0:2*pi/100:2*pi;

            x1 = sqrt(mhdist/a1)*cos(th);
            y1 = sqrt(mhdist/c1)*sin(th);
            
            x = x1*cos(theta) - y1*sin(theta) + cparam(i).mu(1);
            y = x1*sin(theta) + y1*cos(theta) + cparam(i).mu(2);

            % plot the ellipse 
            plot(x,y,color);
        end
    end
end
hold off;
end