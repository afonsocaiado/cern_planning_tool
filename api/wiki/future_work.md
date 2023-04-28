# Future Work on the API

There are several potential improvements and features that can be considered for future development. These enhancements aim to improve the API's performance, user experience, and overall effectiveness in delivering the desired results.

## 1. Explore New Algorithms or Hybrid Techniques to Improve Performance

Investigate and implement novel algorithms or hybrid techniques that could potentially enhance the API's performance. This could include exploring machine learning models, optimization algorithms, or other innovative approaches.

## 2. Periodic Dataset Updates

Implement an API endpoint that periodically updates the activities and contributions datasets. Keeping the datasets up-to-date is crucial to maximize the API's predictive abilities.
This could be done by following the following [document](update_data.md). We would just need to automatically update the csv files in the [orginal_tables](../data/csv_generation/original_tables) folder and then run the [generate_csv.py](../data/csv_generation/generate_csv.py) file.

## 3. Address User Feedback and Improve User Experience

When user feedback is available after deployment or testing, analyze it to identify areas for improvement. Enhance the user experience by addressing the feedback and making necessary adjustments.

## 4. Handle Deployment Issues

Address any issues that arise during deployment after the proof of concept has been completed. This will ensure a smooth transition from development to production.

## 5. Incorporate Facility Names into Suggestions

Incorporate facility names into the suggestions provided by the API. This will give users more context and make the suggestions more informative.
More information on how to do this can be found [here](incorporate_new_field.md). 
Some small adjustments would need to be made, since FACILITY_NAMES is a lits of categorical variables.

## 6. Include Description of "Other" Contributions

When the CONTRIBUTION_TYPE is "Other," include a description of the contribution in the suggestions. This will provide users with more detailed information about the nature of the contribution.
The particularity here is that the Description field is free text, which makes it harder to make suggestions for. One possible approach would simply be to treat it as categorical.
A theoretical approach to address this could be:
1. In the contributions_knn_phase function, separate contributions with the "Other" contribution type and those with other contribution types.
2. For contributions with other contribution types (not "Other"), continue to group them by 'CONTRIBUTION_TYPE' and 'ORG_UNIT_CODE' as you have been doing, and calculate the count of contributions in each group.
3. For contributions with the "Other" contribution type, instead of clustering, simply group them by 'ORG_UNIT_CODE' and 'description'. This will help you identify unique "Other" contributions with different descriptions even if they have the same organization unit.
4. Calculate the count of contributions in each group for the "Other" contributions, similar to how you did for the other contribution types.
5. Combine the results from steps 2 and 4, and find the top common contributions for each phase based on these new groups. This will help you identify the most common and unique contributions by considering the 'CONTRIBUTION_TYPE' and 'ORG_UNIT_CODE', as well as the 'description' field for "Other" contributions.
6. Adjust the confidence calculation to take into account these new counts.

## 7. Additional Filters for Creator Name

When only the creator name is available, consider adding different filters to present more than just the latest activities from that creator. This will offer a more comprehensive view of their contributions and activities.

By implementing these improvements and features, the API will continue to evolve and provide an even better experience for its users.
