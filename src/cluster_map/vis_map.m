% WaterEnergy/cluster_map/vis_map.m

classdef MapPlotter
    properties
        file_path
    end

    methods
        function obj = MapPlotter(file_path)
            % Constructor to initialize the MapPlotter with a file path.
            % Args:
            %     file_path (char): Path to the shapefile.
            obj.file_path = file_path;
        end

        function plot_map(obj, facecolor, edgecolor, figsize)
            % Plot the map using the specified face color, edge color, and figure size.
            % Args:
            %     facecolor (char): Face color of the map.
            %     edgecolor (char): Edge color of the map.
            %     figsize (vector): Figure size [width, height].

            if nargin < 2
                facecolor = 'red';
                edgecolor = 'black';
                figsize = [19.2, 16.8];
            end

            try
                ukraine = readgeotable(obj.file_path); % Read shapefile
                figure('Position', [100, 100, figsize(1)*50, figsize(2)*50]); % Create figure with specified size
                geoshow(ukraine, 'FaceColor', facecolor, 'EdgeColor', edgecolor); % Plot map
                title('Map of Ukraine');
                grid on;
            catch ME
                disp(['An error occurred: ', ME.message]);
            end
        end
    end
end

% Example usage:
% map = MapPlotter('../data/map_data/gadm41_UKR_1.shp');
% map.plot_map();
