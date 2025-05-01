package net.almoney.mcmod.item.custom;

import net.minecraft.world.item.ItemStack;
import net.minecraft.world.item.SwordItem;
import net.minecraft.world.item.Tier;

public class TwilightSword extends SwordItem {

    public TwilightSword(Tier tier, Properties properties) {
        super(tier, properties);
    }

    @Override
    public boolean isFoil(ItemStack stack) {
        return true; // Makes the sword always look enchanted
    }
}
