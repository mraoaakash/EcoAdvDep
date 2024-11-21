# Social isolation reduces metabolic rate and social preference in Wild-type Zebrafish (Danio rerio)

### Abstract
The adaptive and maladaptive effects of social isolation in various animal models, especially under pathogen stress, have generated particular interest since the COVID-19 pandemic. We examined metabolic, social and movement outcomes induced by a month of visual and olfactory social isolation in adult wild-type zebrafish. Our primary interest was examining metabolic savings and changes in social preference, with movement reduction as a manipulation check for effective social isolation. We found that socially isolated fish demonstrated a significant decrease in metabolic rate, social preference and average speed of movement compared to the pre-intervention baseline. The reduction in metabolic rate could lead to contextually beneficial energy expenditure savings for the fish; however, the more mixed effects of reduced social preference and average speed of movement might only be helpful in environments with a high predatory threat and high food and mate abundance. This study opens up various possibilities to explore the adaptive significance of social isolation in various ecological frameworks. 


  **Keywords**: depression, metabolic rate, conspecific social preference, average speed, movement

### Installation Instructions
All enclosed code has been run on a MacbookPro with MacOS Sonoma 14.5 (23F79). The hardware specification is as follows:
- Processor: Apple M1 Max
- Memory: 64 GB
- Graphics: M1 max 32 core
- Storage: 1 TB SSD

The code has been written in Python 3.9.7. The following packages are required to run the code:
- ultralytics
- os
- argparse
- sys
- statsmodels
- tqdm
- cv2
- numpy
- pandas
- matplotlib
- math
- scipy

### Data Availability
All data used in this study has been shared using google drive. The table below highlights the individual files and their respective links.


#### Social Intervention
The YOLOv8 Model trained on the social intervention videos can be found [here](https://drive.google.com/drive/folders/10Bkl8kBM74vEmtIeUuORWO_nc9xfl5-F?usp=drive_link) and the data used to train the model can be found [here](https://drive.google.com/drive/folders/1rutZ2ZI_6izhsX85QmAQ5CMy_p1GOSz9?usp=drive_link).

| File Name             | Pre Data Link | Post Data Link | Description |
| ---                   | --- | --- | --- |
| Resized Videos        | [Pre Int.](https://drive.google.com/drive/folders/1InvCmmiufIhScINKdL5kejQo9ot7DpA1?usp=drive_link) | [Post Int.](https://drive.google.com/drive/folders/1zLIvxjNIOjWiWiNLb-zL8hCB9aiyq_N-?usp=drive_link) | Videos that have been resized to include the central three zones of the tank to measure social preference and other movement parameters.
| Processed Videos      | [Pre Int.](https://drive.google.com/drive/folders/1e0sugPf621fXVNzGQYnUCP9j2ZGpCGSO?usp=drive_link) | [Post Int.](https://drive.google.com/drive/folders/1X4RrokqIBMWsXwkwPwVuUOWNHIx_l6al?usp=drive_link) | Videos that have been converted to black and white and have been shortened to include the experimental phase only.
| Cleaned Videos        | [Pre Int.](https://drive.google.com/open?id=1duvvpnEsKycGZE8ku2AcdTn6OQcB02q8&usp=drive_copy)       | [Post Int.](https://drive.google.com/drive/folders/1_zScuIbrozIQhjblO_HOPopQyb9fCfRF?usp=drive_link) | Videos that have been cleaned to remove any noise and audio.
| Acclam Videos         | [Pre Int.](https://drive.google.com/drive/folders/1sx26zzC5aUNR0-sfsfjIqeCvEukNrgO_?usp=drive_link) | [Post Int.](https://drive.google.com/drive/folders/1qan7uxR_-hQ56Hm_aDevwo2RIzOOqXgF?usp=drive_link) | Videos of the acclamation phase of the experiment.
| Tracked               | [Pre Int.](https://drive.google.com/drive/folders/1RbhYiJA02lqnpEXYHR-KN__x-mWy92o4?usp=drive_link) | [Post Int.](https://drive.google.com/drive/folders/1JyO6fzugw0Ks1DJmbqI1r3b_7IYroRyW?usp=drive_link) | CSV files for each experimental trial video in "Resized Videos" that contain the x and y coordinates and width (w) and height (h) of the fish in each frame found using the YOLOv8 object detection model.
| Metric                | [Pre Int.](https://drive.google.com/drive/folders/13Aaoiieo0spnL71L16kmb0zR3-n1GmSJ?usp=drive_link) | [Post Int.](https://drive.google.com/drive/folders/1OAXHo3IBRwCT2JEZ0p4lQ-EUBBd3DCvZ?usp=drive_link) | Various metrics calculated from the tracked data including the average speed of movement, the distance travelled, and the time spent in each zone of the tank.
| Clean_tracked_centre  | [Pre Int.](https://drive.google.com/drive/folders/1qBUjYO9oeB5g8GdneQ92av2cCdhbZVyg?usp=drive_link) | [Post Int.](https://drive.google.com/drive/folders/1Y6vM7Kt7xB46L-Ct0lPD-Qc5IagPo0z-?usp=drive_link) | CSV files for each experimental trial video in "Resized Videos" that contain the x and y coordinates of the fish in each frame found using the YOLOv8 object detection model, but only for the central three zones of the tank.




### References
Mitsue, S., Yamamoto, T., 2019. Relationship between depression and movement quality in normal young adults. Journal of Physical Therapy Science 31, 819–822. doi:10.1589/jpts.31.819.

Organization, W.H., 2003. Depressive disorder (depression). URL: https://www.who.int/news-room/fact-sheets/detail/depression. 

Young, K.S., Parsons, C.E., Stein, A., Kringelbach, M.L., 2015. Motion and emotion: depression reduces psychomotor performance and alters affective movements in caregiving interactions. Frontiers in Behavioral Neuroscience 9. doi:10.3389/fnbeh.2015.00026.
