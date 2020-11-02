# Projet_3A_IMTA

./Assimilation/

./Detection/
    data/
        data.nc :
            stream data from
            https://resources.marine.copernicus.eu/?option=com_csw&view=details&product_id=GLOBAL_ANALYSIS_FORECAST_PHY_001_024
            temporal resolution: daily-mean
            variables : eastward_sea_water_velocity, northward_sea_water_velocity
            duration : TBD (nb de jours + date du 1er jour)
            area: Arabian sea (52°E-62°E, 13°N-23°N)
    src/
        classes.py :
            StreamLine :representation of a stream line, including the coordinates list and some carateritics.
            Eddy :  representation of an eddy, including the stream line list and some carateritics.
        eddies_detection.py :
            get_traj_with_parcels : compute trajectories of particles in the sea using parcels library, return the list of stream lines.
            get_traj_with_scipy : compute trajectories of particles in the sea using scipy library, return the list of stream lines.
            find_eddies : classify stream line list into eddies.
        plot_tools.py : replace matplotlib and use cartopy for plotting stream lines and eddies.
    tutos/
        eddies_detection_example.ipynb : example of use.
