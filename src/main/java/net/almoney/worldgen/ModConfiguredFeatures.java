package net.almoney.worldgen;

import net.almoney.mcmod.McMod;
import net.almoney.mcmod.block.ModBlocks;
import net.minecraft.core.registries.Registries;
import net.minecraft.data.worldgen.BootstrapContext;
import net.minecraft.resources.ResourceKey;
import net.minecraft.resources.ResourceLocation;
import net.minecraft.tags.BlockTags;
import net.minecraft.world.level.levelgen.feature.ConfiguredFeature;
import net.minecraft.world.level.levelgen.feature.Feature;
import net.minecraft.world.level.levelgen.feature.configurations.FeatureConfiguration;
import net.minecraft.world.level.levelgen.feature.configurations.OreConfiguration;
import net.minecraft.world.level.levelgen.structure.templatesystem.RuleTest;
import net.minecraft.world.level.levelgen.structure.templatesystem.TagMatchTest;

import java.util.List;

public class ModConfiguredFeatures {
    public static final ResourceKey<ConfiguredFeature<?, ?>> OVERWORLD_LURO_ORE_KEY = registerKey("luro_ore");;
    public static final ResourceKey<ConfiguredFeature<?, ?>> OVERWORLD_TWILIGHT_ORE_KEY = registerKey("twilight_ore");;



    public static void bootstrap(BootstrapContext<ConfiguredFeature<?, ?>> context) {
        RuleTest stoneReplaceables = new TagMatchTest(BlockTags.STONE_ORE_REPLACEABLES);
        RuleTest deepslateReplaceables = new TagMatchTest(BlockTags.DEEPSLATE_ORE_REPLACEABLES);


        List<OreConfiguration.TargetBlockState> overworldLuroOres = List.of(
                OreConfiguration.target(stoneReplaceables, ModBlocks.LURO_ORE.get().defaultBlockState()),
                OreConfiguration.target(deepslateReplaceables, ModBlocks.LURO_ORE.get().defaultBlockState()));

        List<OreConfiguration.TargetBlockState> overworldTwilightOres = List.of(
                OreConfiguration.target(deepslateReplaceables, ModBlocks.TWILIGHT_ORE.get().defaultBlockState()),
                OreConfiguration.target(stoneReplaceables, ModBlocks.TWILIGHT_ORE.get().defaultBlockState()));



        register(context, OVERWORLD_LURO_ORE_KEY, Feature.ORE, new OreConfiguration(overworldLuroOres, 9));
        register(context, OVERWORLD_TWILIGHT_ORE_KEY, Feature.ORE, new OreConfiguration(overworldTwilightOres, 5));

    }

    public static ResourceKey<ConfiguredFeature<?, ?>> registerKey(String name) {
        return ResourceKey.create(Registries.CONFIGURED_FEATURE, ResourceLocation.fromNamespaceAndPath(McMod.MOD_ID, name));
    }

    private static <FC extends FeatureConfiguration, F extends Feature<FC>> void register(BootstrapContext<ConfiguredFeature<?, ?>> context,
                                                                                          ResourceKey<ConfiguredFeature<?, ?>> key, F feature, FC configuration) {
        context.register(key, new ConfiguredFeature<>(feature, configuration));
    }
}