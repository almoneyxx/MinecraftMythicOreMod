package net.almoney.mcmod.item;

import net.almoney.mcmod.McMod;
import net.minecraft.world.item.Item;
import net.minecraft.world.item.SwordItem;
import net.minecraftforge.eventbus.api.IEventBus;
import net.minecraftforge.registries.DeferredRegister;
import net.minecraftforge.registries.ForgeRegistries;
import net.minecraftforge.registries.RegistryObject;

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
            () -> new SwordItem(ModToolTiers.TWILIGHT, new Item.Properties()
                    .attributes(SwordItem.createAttributes(ModToolTiers.TWILIGHT, 4, -2.0F))
            ));





    public static void register(IEventBus eventBus) {
        ITEMS.register(eventBus);

    }
}
