# Stream Slope Methods

ARC provides several methods for estimating stream slope at either the reach level or the cell level. These methods include:

1. **Local Average**: This method finds all stream cells within a specified number of cells (`Gen_Slope_Dist`) of the target stream cell, and averages the slopes between those cells and the target cell. Activated by setting `Stream_Slope_Method` to "local_average".
2. **Local Average Corrected**: This method is similar to the local average method, but forces the slope to be within the inter-quartile range of slopes, computed across all cells in the reach. Activated by setting `Stream_Slope_Method` to "local_average_corrected".
3. **Reach Average**: This method uses the provided stream vector file (usually a shapefile) to estimate the slope from the endpoints of the stream reach. This value is then used for each stream cell in the reach. Activated by setting `Stream_Slope_Method` to "reach_average" or "end_points". Requires `StrmShp_File` to be defined.