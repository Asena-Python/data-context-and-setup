
import pandas as pd
import numpy as np
from olist.utils import haversine_distance
from olist.data import Olist


class Order:
    """
    DataFrames containing all orders as index,
    and various properties of these orders as columns
    """

    def __init__(self):
        self.data = Olist().get_data()

    def get_wait_time(self, is_delivered=True):
        """
        Returns a DataFrame with:
        [order_id, wait_time, expected_wait_time,
        delay_vs_expected, order_status]
        """
        orders = self.data['orders'].copy()

        if is_delivered:
            orders = orders[
                orders['order_status'] == 'delivered'
            ].copy()

        date_columns = [
            'order_purchase_timestamp',
            'order_delivered_customer_date',
            'order_estimated_delivery_date'
        ]

        for col in date_columns:
            orders[col] = pd.to_datetime(orders[col])

        orders['wait_time'] = (
            orders['order_delivered_customer_date']
            - orders['order_purchase_timestamp']
        ).dt.total_seconds() / 86400

        orders['expected_wait_time'] = (
            orders['order_estimated_delivery_date']
            - orders['order_purchase_timestamp']
        ).dt.total_seconds() / 86400

        orders['delay_vs_expected'] = (
            orders['wait_time'] - orders['expected_wait_time']
        ).clip(lower=0)

        return orders[
            [
                'order_id',
                'wait_time',
                'expected_wait_time',
                'delay_vs_expected',
                'order_status'
            ]
        ]

    def get_review_score(self):
        """
        Returns a DataFrame with:
        order_id, dim_is_five_star, dim_is_one_star, review_score
        """
        reviews = self.data['order_reviews'].copy()

        reviews['dim_is_five_star'] = (
            reviews['review_score'] == 5
        ).astype(int)

        reviews['dim_is_one_star'] = (
            reviews['review_score'] == 1
        ).astype(int)

        return reviews[
            [
                'order_id',
                'dim_is_five_star',
                'dim_is_one_star',
                'review_score'
            ]
        ].copy()

    def get_number_items(self):
        """
        Returns a DataFrame with:
        order_id, number_of_items
        """
        items = self.data['order_items'].copy()

        number_items_df = (
            items.groupby('order_id')
            .size()
            .reset_index(name='number_of_items')
        )

        return number_items_df

    def get_number_sellers(self):
        """
        Returns a DataFrame with:
        order_id, number_of_sellers
        """
        items = self.data['order_items'].copy()

        number_sellers_df = (
            items.groupby('order_id')['seller_id']
            .nunique()
            .reset_index(name='number_of_sellers')
        )

        return number_sellers_df

    def get_price_and_freight(self):
        """
        Returns a DataFrame with:
        order_id, price, freight_value
        """
        items = self.data['order_items'].copy()

        price_freight_df = (
            items.groupby('order_id')[
                ['price', 'freight_value']
            ]
            .sum()
            .reset_index()
        )

        return price_freight_df

    # Optional
    def get_distance_seller_customer(self):
                    """
                    Returns a DataFrame with:
                    order_id, distance_seller_customer
                    """

                    orders = self.data['orders']
                    order_items = self.data['order_items']
                    sellers = self.data['sellers']
                    customers = self.data['customers']
                    geolocation = self.data['geolocation']

                    matching_geo = orders.merge(
                        customers,
                        on='customer_id',
                        how='left'
                    )

                    matching_geo = matching_geo.merge(
                        order_items[['order_id', 'seller_id']],
                        on='order_id',
                        how='left'
                    )

                    matching_geo = matching_geo.merge(
                        sellers,
                        on='seller_id',
                        how='left'
                    )

                    geo_avg = (
                        geolocation.groupby('geolocation_zip_code_prefix')[
                            ['geolocation_lat', 'geolocation_lng']
                        ]
                        .mean()
                        .reset_index()
                    )

                    matching_geo = matching_geo.merge(
                        geo_avg,
                        left_on='customer_zip_code_prefix',
                        right_on='geolocation_zip_code_prefix',
                        how='left'
                    )

                    matching_geo = matching_geo.rename(columns={
                        'geolocation_lat': 'customer_lat',
                        'geolocation_lng': 'customer_lng'
                    })

                    matching_geo = matching_geo.merge(
                        geo_avg,
                        left_on='seller_zip_code_prefix',
                        right_on='geolocation_zip_code_prefix',
                        how='left',
                        suffixes=('', '_seller')
                    )

                    matching_geo = matching_geo.rename(columns={
                        'geolocation_lat': 'seller_lat',
                        'geolocation_lng': 'seller_lng'
                    })
                    import numpy as np

                    lat1 = np.radians(matching_geo['customer_lat'])
                    lon1 = np.radians(matching_geo['customer_lng'])
                    lat2 = np.radians(matching_geo['seller_lat'])
                    lon2 = np.radians(matching_geo['seller_lng'])

                    dlat = lat2 - lat1
                    dlon = lon2 - lon1

                    a = (
                        np.sin(dlat / 2) ** 2
                        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
                    )

                    matching_geo['distance_seller_customer'] = (
                        6371 * 2 * np.arcsin(np.sqrt(a.clip(0, 1)))
                    )
                    matching_geo = (
                        matching_geo.groupby('order_id', as_index=False)
                        ['distance_seller_customer']
                        .mean()
                    )

                    return matching_geo

    def get_training_data(
        self,
        is_delivered=True,
        with_distance_seller_customer=False
    ):
        """
        Returns a clean DataFrame without NaN values.
        """

        training_df = self.get_wait_time(
            is_delivered=is_delivered
        )

        training_df = training_df.merge(
            self.get_review_score(),
            on='order_id',
            how='left'
        )

        training_df = training_df.merge(
            self.get_number_items(),
            on='order_id',
            how='left'
        )

        training_df = training_df.merge(
            self.get_number_sellers(),
            on='order_id',
            how='left'
        )

        training_df = training_df.merge(
            self.get_price_and_freight(),
            on='order_id',
            how='left'
        )

        if with_distance_seller_customer:
            training_df = training_df.merge(
                self.get_distance_seller_customer(),
                on='order_id',
                how='left'
            )

        training_df = training_df.dropna().copy()

        return training_df
