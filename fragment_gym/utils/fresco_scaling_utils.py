#!/usr/bin/python3

from shapely import (
    affinity,
    MultiPolygon,
    MultiPoint
)

class FrescoScalingUtils():
    def __init__(self, config):
        self.config = config
    
    def scale_fresco(self, fresco_polygons, scale_factor):
        # Scale up fresco
        sf = 1.0 + scale_factor
        scaled_polygons = affinity.scale(
            fresco_polygons, sf, sf
        )

        # Shift old fragments according to new centroids
        new_polygons = []
        new_centroids = []
        for i in range(len(fresco_polygons.geoms)):
            old_centroid = fresco_polygons.geoms[i].centroid
            new_centroid = scaled_polygons.geoms[i].centroid

            shifted_fragment = affinity.translate(
                fresco_polygons.geoms[i],
                new_centroid.x-old_centroid.x,
                new_centroid.y-old_centroid.y,
            )

            new_polygons.append(shifted_fragment)
            new_centroids.append(new_centroid)

        # group all polygons into a shapely multipolygon
        fresco_polygons_scaled = MultiPolygon(new_polygons)
        fresco_centroids_scaled = MultiPoint(new_centroids)

        return fresco_polygons_scaled, fresco_centroids_scaled