import pandas as pd
import numpy as np
import dictionary as reference
import seaborn as sns
import time
from shapely.geometry import Point
from ema_workbench import (MultiprocessingEvaluator)
import math
import ema_workbench
from ema_workbench import (Model, Constant, RealParameter, ScalarOutcome, ema_logging,
                           perform_experiments)
import matplotlib.pyplot as plt
import numba
import warnings

warnings.filterwarnings(action='once')


def add_nearest_Points(df):
    df['geometry'] = df.apply(lambda z: Point(z.Longitude, z.Latitude), axis=1)

    def near(row):
        # find the nearest point and return the corresponding Place value
        distance = []

        [distance.append({'distance': row.geometry.distance(d), 'Parcel_ID': p}) for d, p in row.range if
         not pd.isnull(d)]

        sorted_distance = sorted(distance, key=lambda k: k['distance'])
        return sorted_distance[1:9]

    df = df.sort_values(by=["Latitude"])

    for i in range(-8, 9):
        df['x_range_{}'.format(i)] = list(zip(df['geometry'].shift(i), df.Parcel_ID.shift(i)))

    df = df.sort_values(by=["Longitude"])

    for i in range(-8, 9):
        df['y_range_{}'.format(i)] = list(zip(df['geometry'].shift(i), df.Parcel_ID.shift(i)))

    def changed(x):
        layer = []
        for i in range(-8, 9):
            if i != 0:
                if not pd.isnull(x['x_range_{}'.format(i)]):
                    layer.append(x['x_range_{}'.format(i)])
                if not pd.isnull(x['y_range_{}'.format(i)]):
                    layer.append(x['y_range_{}'.format(i)])
        return layer

    df['range'] = df.apply(lambda x: changed(x), axis=1)

    def x_s(i):
        return 'x_range_{}'.format(i)

    def y_s(i):
        return 'y_range_{}'.format(i)

    # column.append(['y_range_{}'.format(i)for i in range(-8,9) if i != 0])

    df['Nearest'] = df.apply(lambda row: near(row), axis=1)

    df = df.drop([x_s(i) for i in range(-8, 9) if i != 0], axis=1)
    df = df.drop([y_s(i) for i in range(-8, 9) if i != 0], axis=1)
    # print(df)
    return df


def elevation(Foundation, Bldg_Area, flood_depth, adjustment):
    '''

Equation to estimate elevation cost
---
elevation = Bldg_Area * Foundation

    :param Foundation: Foundation Type
    :param Bldg_Area:  Square foot area
    :param flood_depth: Flood Depth Estimated from storm surge or inland flood.
    :return: Elevation cost for a parcel.
    '''
    if flood_depth > 5:
        extra_elevation = 5
    else:
        extra_elevation = flood_depth
    return Bldg_Area * (
            reference.Foundations(Foundation) + reference.extra_perfoot_Raised(extra_elevation)) * adjustment


@numba.jit()
def extra_elevation(flood_depth):
    '''

     :param Foundation: Foundation Type
    :param Bldg_Area:  Square foot area
    :param flood_depth: Flood Depth Estimated from storm surge or inland flood.
    :return: Elevation needed to reduce flooding.
    '''
    if flood_depth > 5:
        extra_elevation = 5
    else:
        extra_elevation = flood_depth
    return extra_elevation


# @numba.jit()
def inundated(x, y):
    try:
        if x - y > 0:
            return True
        else:
            return False
    except:
        return False


def REL(Quality, Exterior, Imp_Type, Improvement_Value, Replacement_Cost):
    '''

     Calculate Remaining Expected Life REL
    :param Quality: Building QUality pandas column Bldg_Quali
    :param Exterior: Exterior Type pandas column Exterior_F
    :param Imp_Type: Improvement type pandas column Imp_Code
    :param Improvement_Value: Structure Improvement Value
    :param Replacement_Cost:   Estimated Cost to replace structure.
    :return:
    '''
    return (reference.economic_life(Quality, Imp_Type) + reference.exterior_addValue(Exterior)) * (
            Improvement_Value / Replacement_Cost)


# @numba.jit()
def discount(aal, REL):
    '''
     Buyout Savings Discounted AAL
    calculate future savings using a discount rate
    See FEMA  BCA Library, appendix B-5
    :param aal: Average Annualized Loses
    :param REL: Remaining Expected life Expectancy.
    :return: discounted savings.
    '''
    discount_rate = 0.03
    Year_of_Buyout = 0
    savings = 0
    for i in range(0, round(int(REL))):
        try:
            powd = pow((1 + discount_rate), (i))
        except:
            # print(REL)
            exit(31)
        savings += (aal - 550) / powd
    return savings


def load_clean_data(csv, columns, ):
    df = pd.read_csv(csv)

    '''Column variables for each dataframe. Main Differences are the 
     Ike_Base, and Harvey_Depth Variables. These Variables differ by type of flooding.
     Fill nan values'''
    print(len(df))
    df = df[columns]
    df.drop_duplicates("Parcel_ID", keep="last", inplace=True)
    df.drop_duplicates(["Latitude","Longitude"], keep="last", inplace=True)
    df.fillna(0, inplace=True)
    '''Create a function to test for inundation. 
    Uses Assumed_FF which is the first floor elevation, compared with the flood depth'''

    ''' Calculate whether a parcel was inundated. '''
    df['Inundated_Base'] = inundated(df['Total_Depth'], df["Assumed_FF"])

    df.sort_values(by=['Total_Depth'], inplace=True)

    print(len(df))
    df = add_nearest_Points(df)

    df.set_index("Parcel_ID", inplace=True)  # Set index for joins

    ''' Load Claims dataset.
        Clean duplicates and set index.
        Join to SS and IF
    '''
    df_claims = pd.read_csv("claims.csv")  # claims dataset to join to parcels
    df_claims.drop_duplicates("Parcel_ID", keep="last", inplace=True)
    df_claims.set_index("Parcel_ID", inplace=True)

    result = pd.concat([df, df_claims], axis=1, join_axes=[df.index])
    print(len(result))
    result = result.drop(columns=["OBJECTID"])

    # result['extra_elevation_SS'] = result.apply(lambda x: extra_elevation(x['Ike_Base']), axis=1)
    result['extra_elevation_IF'] = result.apply(lambda x: extra_elevation(x['Total_Depth']), axis=1)

    '''Update Residential Curves '''

    result.State_Code = reference.Residential_Code(result.State_Code)

    result.reset_index(inplace=True)
    '''Fill nan again'''

    result.fillna(0, inplace=True)
    # result["AAL_SS"] = abs(result.AAL_SS)
    result["AAL_IF"] = abs(result.AAL_IF)
    print(len(result))
    return result


def return_BCR(Savings, Cost):
    #     if Cost == 0 or Savings==Savings: return 0
    if Savings / Cost != np.inf:
        return Savings / Cost
    else:
        return 0


def weighted_BCR(df, result_ss):
    weighted_Parcels = [int(i["Parcel_ID"]) for i in df.Nearest]

    return (df.Buyout_Savings + result_ss[result_ss.Parcel_ID.isin(weighted_Parcels)].Buyout_Savings.sum()) / (
            df.Buyouts_Cost + result_ss[result_ss.Parcel_ID.isin(weighted_Parcels)].Buyouts_Cost.sum())


def calculate_uncertainty_aspects(result_ss, flood_type, market_value_adjuster, demolition=7,
                                  elevation_cost_adjustment=1, budget=750, ratio=5, weighted_bcr_flag=True):
    '''

    :param demolition: Demolition cost per square foot
    :param cap:

    Read CSV Files into pandas dataframes.
    -------------------------------------
    Dataframe acronyms:

    SS = Storm Surge Parcels

    IF = Inland Flooding

    return Dataframe
    '''

    '''Calculates the Buyout cost
    # ---
    # buyout = Market_Value + Bldg_Area

    '''
    if flood_type == "IF":
        depth_column = 'Total_Depth'

    else:
        depth_column = "Harvey_Dep"
    result_ss = result_ss[result_ss[depth_column] > 0]
    result_ss['Buyouts_Cost'] = result_ss.Imp_Value + result_ss.Land_Value * market_value_adjuster + (
            result_ss['Bldg_Area'] * demolition)

    result_ss['REL'] = result_ss.apply(lambda x: REL(x.Bldg_Quali, x.Exterior_F, x.Imp_Code,
                                                     x.Imp_Value,
                                                     (x.Imp_Value + x.Land_Value * market_value_adjuster + .001)),
                                       axis=1)

    result_ss["Buyout_AAL_Savings"] = result_ss.apply(lambda x: discount(x['AAL_{}'.format(flood_type)], x.REL), axis=1)

    result_ss["Buyout_Savings"] = result_ss.All_Claims_PV + result_ss["Buyout_AAL_Savings"]

    if weighted_bcr_flag == True:

        result_ss["Buyouts_BCR"] = result_ss.apply(lambda x: weighted_BCR(x, result_ss), axis=1)

    else:

        result_ss["Buyouts_BCR"] = (result_ss["Buyout_Savings"]) / result_ss["Buyouts_Cost"]

    ''' Elevation Cost '''
    result_ss['Elevation_Cost_1ft'] = result_ss.apply(lambda x: elevation(x['Foundation'], x['Bldg_Area'],
                                                                          int(round(x[depth_column])), elevation_cost_adjustment),
                                                      axis=1)

    ''' Estimated damages with and without Elevation'''
    start = time.time()
    result_ss["Elevation_Est_Damage"] = result_ss.apply(lambda x: RES_damage(x.State_Code, x.Storey, damage_curves,
                                                                             x.Imp_Value * market_value_adjuster,
                                                                             (int(round(x[depth_column])) - x.Assumed_FF - x[
                                                                                 'extra_elevation_{}'.format(
                                                                                     flood_type)])), axis=1)
    #     print(time.time() - start)
    result_ss["Est_Damage"] = result_ss.apply(lambda x: RES_damage(x.State_Code, x.Storey, damage_curves,
                                                                   x.Imp_Value * market_value_adjuster,
                                                                   (int(round(x[depth_column])) - x.Assumed_FF)), axis=1)

    '''Estimate Savings'''

    result_ss["Elevation_Savings"] = result_ss["Est_Damage"] - result_ss["Elevation_Est_Damage"]

    '''Estimate BCR'''

    result_ss['Elevation_BCR'] = (result_ss.Elevation_Savings) / result_ss.Elevation_Cost_1ft

    buyout_ratio = ratio / 10
    '''Buyout Cap'''
    result_ss = result_ss.sort_values('Buyouts_BCR', ascending=False)

    result_ss['Buyout_Rolling'] = result_ss['Buyouts_Cost'][
        (result_ss.Buyouts_BCR > 0.75) & (result_ss.Market_Val > 0)].rolling(len(result_ss), min_periods=1).sum()

    result_ss = result_ss.sort_values('Elevation_BCR', ascending=False)
    elevationBudget_ss = result_ss[
        ['Elevation_Savings', 'Elevation_Cost_1ft', 'FIPS', 'Buyout_Savings', 'Buyouts_Cost', 'Buyouts_BCR',
         'Elevation_BCR']]

    elevationBudget_ss['Elevation_Rolling'] = elevationBudget_ss['Elevation_Cost_1ft'][
        (result_ss.Elevation_BCR > 0.75) & (result_ss.Market_Val > 0)].rolling(len(result_ss),
                                                                               min_periods=1).sum()

    '''Run Budgets for mitigation practices'''

    # buyouts

    BuyoutBudget_ss = elevationBudget_ss[['Buyout_Savings', 'Buyouts_Cost', 'Buyouts_BCR', 'FIPS']][
        result_ss.Buyout_Rolling < buyout_ratio * budget * 1000000]

    # Elevation

    elevationBudget_ss = elevationBudget_ss[['Elevation_Savings', 'Elevation_Cost_1ft', 'Elevation_BCR', 'FIPS']][
        (elevationBudget_ss.Elevation_Rolling < ((1 - buyout_ratio) * budget * 1000000))]

    elevation_bcr = return_BCR(elevationBudget_ss['Elevation_Savings'].sum(),
                               elevationBudget_ss['Elevation_Cost_1ft'].sum())
    buyout_bcr = return_BCR(BuyoutBudget_ss['Buyout_Savings'].sum(), BuyoutBudget_ss['Buyouts_Cost'].sum())
    if elevation_bcr == np.nan: elevation_bcr = 0
    return len(BuyoutBudget_ss), len(elevationBudget_ss), BuyoutBudget_ss['Buyout_Savings'].sum(), BuyoutBudget_ss[
        'Buyouts_Cost'].sum(), buyout_bcr, elevationBudget_ss['Elevation_Savings'].sum(), elevationBudget_ss[
               'Elevation_Cost_1ft'].sum(), elevation_bcr, BuyoutBudget_ss['Buyout_Savings'].sum() + elevationBudget_ss[
               'Elevation_Savings'].sum(), BuyoutBudget_ss['Buyouts_Cost'].sum() + elevationBudget_ss[
               'Elevation_Cost_1ft'].sum(), BuyoutBudget_ss['Buyout_Savings'].sum() + elevationBudget_ss[
               'Elevation_Savings'].sum() / BuyoutBudget_ss['Buyouts_Cost'].sum() + elevationBudget_ss[
               'Elevation_Cost_1ft'].sum()


def output_UncertainAspects(result_ss, flood_type, market_value_adjuster, demolition=7,
                            elevation_cost_adjustment=1, budget=750, ratio=5, weighted_bcr_flag=True, outfile="test.csv"):
    '''

      :param demolition: Demolition cost per square foot
      :param cap:

      Read CSV Files into pandas dataframes.
      -------------------------------------
      Dataframe acronyms:

      SS = Storm Surge Parcels

      IF = Inland Flooding

      return Dataframe
      '''

    '''Calculates the Buyout cost
    # ---
    # buyout = Market_Value + Bldg_Area

    '''
    if flood_type == "IF":
        depth_column = 'Total_Depth'

    else:
        depth_column = "Harvey_Dep"
    result_ss = result_ss[result_ss[depth_column] > 0]
    result_ss['Buyouts_Cost'] = result_ss.Imp_Value + result_ss.Land_Value * market_value_adjuster + (
            result_ss['Bldg_Area'] * demolition)

    result_ss['REL'] = result_ss.apply(lambda x: REL(x.Bldg_Quali, x.Exterior_F, x.Imp_Code,
                                                     x.Imp_Value,
                                                     (x.Imp_Value + x.Land_Value * market_value_adjuster + .001)),
                                       axis=1)

    result_ss["Buyout_AAL_Savings"] = result_ss.apply(lambda x: discount(x['AAL_{}'.format(flood_type)], x.REL), axis=1)

    result_ss["Buyout_Savings"] = result_ss.All_Claims_PV + result_ss["Buyout_AAL_Savings"]

    if weighted_bcr_flag == True:

        result_ss["Buyouts_BCR"] = result_ss.apply(lambda x: weighted_BCR(x, result_ss), axis=1)

    else:

        result_ss["Buyouts_BCR"] = (result_ss["Buyout_Savings"]) / result_ss["Buyouts_Cost"]

    ''' Elevation Cost '''
    result_ss['Elevation_Cost_1ft'] = result_ss.apply(lambda x: elevation(x['Foundation'], x['Bldg_Area'],
                                                                          int(round(x[depth_column])),
                                                                          elevation_cost_adjustment),
                                                      axis=1)

    ''' Estimated damages with and without Elevation'''
    start = time.time()
    result_ss["Elevation_Est_Damage"] = result_ss.apply(lambda x: RES_damage(x.State_Code, x.Storey, damage_curves,
                                                                             x.Imp_Value * market_value_adjuster,
                                                                             (int(round(
                                                                                 x[depth_column])) - x.Assumed_FF - x[
                                                                                  'extra_elevation_{}'.format(
                                                                                      flood_type)])), axis=1)
    #     print(time.time() - start)
    result_ss["Est_Damage"] = result_ss.apply(lambda x: RES_damage(x.State_Code, x.Storey, damage_curves,
                                                                   x.Imp_Value * market_value_adjuster,
                                                                   (int(round(x[depth_column])) - x.Assumed_FF)),
                                              axis=1)

    '''Estimate Savings'''

    result_ss["Elevation_Savings"] = result_ss["Est_Damage"] - result_ss["Elevation_Est_Damage"]

    '''Estimate BCR'''

    result_ss['Elevation_BCR'] = (result_ss.Elevation_Savings) / result_ss.Elevation_Cost_1ft

    buyout_ratio = ratio / 10
    '''Buyout Cap'''
    result_ss = result_ss.sort_values('Buyouts_BCR', ascending=False)

    result_ss['Buyout_Rolling'] = result_ss['Buyouts_Cost'][
        (result_ss.Buyouts_BCR > 0.75) & (result_ss.Market_Val > 0)].rolling(len(result_ss), min_periods=1).sum()

    result_ss = result_ss.sort_values('Elevation_BCR', ascending=False)
    elevationBudget_ss = result_ss
    # [
    #     ['Elevation_Savings', 'Elevation_Cost_1ft', 'FIPS', 'Buyout_Savings', 'Buyouts_Cost', 'Buyouts_BCR','Buyout_Rolling',
    #      'Elevation_BCR', 'Longitude', 'Latitude']]

    elevationBudget_ss['Elevation_Rolling'] = elevationBudget_ss['Elevation_Cost_1ft'][
        (result_ss.Elevation_BCR > 0.75) & (result_ss.Market_Val > 0)].rolling(len(result_ss),
                                                                               min_periods=1).sum()

    elevationBudget_ss.to_csv(outfile)
    return elevationBudget_ss


class DamageCurves:
    """Class to import damage curves into pandas damage curves.
        curves are broken into residential and commercial curves for both building and content.
        commercial curves include all landuses not considered residential """

    def load_curves(self, residential_curve="Residential.damages", residential_content="ResidentialContent.damages"):
        '''

        :param residential_curve: dataframe with residential curves
        :param residential_content: dataframe with estimated residential content curves
        :return:
        '''
        self.residential_curve_df = pd.read_csv(residential_curve)
        self.residential_content_df = pd.read_csv(residential_content)


def RES_damage(residential_Type, stories_values, damage_curves, structure_values,
               inundation):
    '''

    :param residential_Type:  Type of Residential Property, Single Family (res1), etc.
    :param stories_values: Number of stories for a home
    :param damage_curves: Location of Damage Curves.
    :param structure_values: Estiamted Value of the Structure
    :param inundation:  Estimated flood indundation for  the residential property.
    :return: Estimated damages.
    '''
    #     stories_values = np.where(stories_values > 2, 2, 1)
    if stories_values > 2: stories_values = 2
    if stories_values == 0: stories_values = 1
    #     inundation = np.where(inundation<= 0, 0, np.where(inundation ==np.nan, 0, np.where( inundation> 24, 24, round(inundation))))
    if inundation <= 0 or inundation == np.nan: return 0
    if inundation > 24: inundation = 24
    #     inundation =
    #     res_column = np.where(residential_Type=='RES1',residential_Type + "_" + str(int(stories_values)), residential_Type )
    if residential_Type == 'RES1':
        res_column = residential_Type + "_" + str(int(stories_values))
    else:
        res_column = residential_Type

    res_list = list(damage_curves.residential_curve_df.columns.values)
    try:
        damage_percent = damage_curves.residential_curve_df[res_column].loc[inundation]
        content_percent = damage_curves.residential_content_df[res_column].loc[inundation]
    except:
        return 0

    #         print(damage_percent, content_percent)
    output = ((structure_values * (damage_percent)) + (0.5 * structure_values * (content_percent)))
    #         print(output)
    return output


damage_curves = DamageCurves()
damage_curves.load_curves()

damage_curves.residential_content_df.Inundation = (damage_curves.residential_content_df.Inundation * 3.28084).astype(
    int)
damage_curves.residential_curve_df.Inundation = (damage_curves.residential_curve_df.Inundation * 3.28084).astype(int)

if __name__=="__main__":
    start = time.time()
    print("start")
    ss_Columns = ['Parcel_ID', "Foundation", "Imp_Value", 'Imp_Code', 'Bldg_Quali', "FLD_ZONE", "TotalRepla",
                  "Year_Built",
                  "Exterior_F", "Latitude",
                  "Longitude", "Market_Val", 'State_Code', 'County', "Storey", "Bldg_Area", "Assumed_FF", "Land_Value",
                  'FIPS', 'AAL_IF', 'AAL_SS', 'Harvey_Dep', 'Ike_Base', "Total_Depth"]
    result_ss = load_clean_data("IF_SS.CSV", ss_Columns)
    # result_if = load_clean_data("Spatial_Parcel_AAL_if2.csv", if_Columns, "Harvey_Depth")
    print((time.time() - start) / 60)
    result_ss = output_UncertainAspects(result_ss, 'SS',
                                          market_value_adjuster=0.90722791,
                                          demolition=8.09490431,
                                          elevation_cost_adjustment=1.222582,
                                          weighted_bcr_flag=True)
