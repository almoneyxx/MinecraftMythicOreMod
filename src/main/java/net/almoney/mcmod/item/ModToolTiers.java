package net.almoney.mcmod.item;

import net.almoney.mcmod.item.ModItems;
import net.minecraft.tags.BlockTags;
import net.minecraft.world.item.Tier;
import net.minecraft.world.item.crafting.Ingredient;
import net.minecraftforge.common.ForgeTier;

public class ModToolTiers {
    public static final Tier TWILIGHT = new ForgeTier(
            1800,              //durability
            9.0f,              // speed mining speed
            3.0f,              // attackDamageBonus
            15,                // enchantment value
            BlockTags.NEEDS_DIAMOND_TOOL,          //mineable blocks
            () -> Ingredient.of(ModItems.TWILIGHTINGOT.get()), // repair
            null
    );
}

