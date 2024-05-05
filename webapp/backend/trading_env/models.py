from django.db import models

class FullChannel(models.Model):
    order_id = models.CharField(max_length=100)
    order_type = models.CharField(max_length=20, blank=True, null=True)
    size = models.FloatField(blank=True, null=True)
    price = models.FloatField()
    client_oid = models.CharField(max_length=100, blank=True, null=True)
    type = models.CharField(max_length=20)
    side = models.CharField(max_length=10)
    product_id = models.CharField(max_length=50)
    time = models.DateTimeField()
    sequence = models.BigIntegerField(blank=True, null=True)
    remaining_size = models.FloatField(blank=True, null=True)
    trade_id = models.BigIntegerField(blank=True, null=True)
    maker_order_id = models.CharField(max_length=100, blank=True, null=True)
    taker_order_id = models.CharField(max_length=100, blank=True, null=True)
    reason = models.CharField(max_length=50, blank=True, null=True)
    funds = models.FloatField(blank=True, null=True)
    old_size = models.FloatField(blank=True, null=True)
    new_size = models.FloatField(blank=True, null=True)

    def __str__(self):
        return f"{self.product_id} - {self.side} - {self.price}"

class Ticker(models.Model):
    type = models.CharField(max_length=20)
    sequence = models.BigIntegerField()
    product_id = models.CharField(max_length=50)
    price = models.FloatField()
    open_24h = models.FloatField()
    volume_24h = models.FloatField()
    low_24h = models.FloatField()
    high_24h = models.FloatField()
    volume_30d = models.FloatField()
    best_bid = models.FloatField()
    best_ask = models.FloatField()
    side = models.CharField(max_length=10)
    time = models.DateTimeField()
    trade_id = models.BigIntegerField()
    last_size = models.FloatField()

    def __str__(self):
        return f"{self.product_id} - {self.price} - {self.time}"

