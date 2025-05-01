package net.almoney.mcmod.item;

import net.almoney.mcmod.McMod;
import net.minecraft.world.item.ArmorItem;
import net.minecraft.world.item.Item;
import net.minecraft.world.item.SwordItem;
import net.minecraftforge.eventbus.api.IEventBus;
import net.minecraftforge.registries.DeferredRegister;
import net.minecraftforge.registries.ForgeRegistries;
import net.minecraftforge.registries.RegistryObject;
import net.almoney.mcmod.item.custom.TwilightSword;

public class ModItems {
    public static final DeferredRegister<Item> ITEMS =
            DeferredRegister.create(ForgeRegistries.ITEMS, McMod.MOD_ID);

    public static final RegistryObject<Item> TWILIGHTINGOT  = ITEMS.register("twilightingot",
            () -> new Item(new Item.Properties()));

    public static final RegistryObject<Item> LUROINGOT = ITEMS.register("luroingot",
            () -> new Item(new Item.Properties()));

    public static final RegistryObject<Item> RAWLURO = ITEMS.register("rawluro",
            () -> new Item(new Item.Properties()));

    public static final RegistryObject<Item> RAWTWILIGHT = ITEMS.register("rawtwilight",
            () -> new Item(new Item.Properties()));

    public static final RegistryObject<Item> TWILIGHT_SWORD = ITEMS.register("twilight_sword",
            () -> new TwilightSword(ModToolTiers.TWILIGHT, new Item.Properties()
                    .attributes(SwordItem.createAttributes(ModToolTiers.TWILIGHT, 4, -2.0F))
            ));

    public static final RegistryObject<Item> LURO_HELMET = ITEMS.register("luro_helmet",
            () -> new ArmorItem(ModArmorMaterials.LURO_ARMOR_MATERIAL, ArmorItem.Type.HELMET,
                    new Item.Properties().durability(ArmorItem.Type.HELMET.getDurability(22))));

    public static final RegistryObject<Item> LURO_CHESTPLATE = ITEMS.register("luro_chestplate",
            () -> new ArmorItem(ModArmorMaterials.LURO_ARMOR_MATERIAL, ArmorItem.Type.CHESTPLATE,
                    new Item.Properties().durability(ArmorItem.Type.CHESTPLATE.getDurability(22))));

    public static final RegistryObject<Item> LURO_LEGGINGS = ITEMS.register("luro_leggings",
            () -> new ArmorItem(ModArmorMaterials.LURO_ARMOR_MATERIAL, ArmorItem.Type.LEGGINGS,
                    new Item.Properties().durability(ArmorItem.Type.LEGGINGS.getDurability(22))));

    public static final RegistryObject<Item> LURO_BOOTS = ITEMS.register("luro_boots",
            () -> new ArmorItem(ModArmorMaterials.LURO_ARMOR_MATERIAL, ArmorItem.Type.BOOTS,
                    new Item.Properties().durability(ArmorItem.Type.BOOTS.getDurability(22))));





    public static void register(IEventBus eventBus) {
        ITEMS.register(eventBus);

    }
}
