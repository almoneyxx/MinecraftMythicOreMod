package net.almoney.mcmod.event;

import net.almoney.mcmod.McMod;
import net.almoney.mcmod.item.ModItems;
import net.minecraft.core.particles.ParticleTypes;
import net.minecraft.server.level.ServerLevel;
import net.minecraft.world.entity.player.Player;
import net.minecraft.world.item.ItemStack;
import net.minecraft.world.phys.Vec3;
import net.minecraftforge.api.distmarker.Dist;
import net.minecraftforge.event.entity.living.LivingAttackEvent;
import net.minecraftforge.eventbus.api.SubscribeEvent;
import net.minecraftforge.fml.common.Mod;


@Mod.EventBusSubscriber(modid = McMod.MOD_ID, bus = Mod.EventBusSubscriber.Bus.FORGE, value = Dist.CLIENT)
public class ModEvents {

    @SubscribeEvent
    public static void onTwilightSwordHit(LivingAttackEvent event) {
        if (!(event.getSource().getEntity() instanceof Player player)) return;

        ItemStack mainHand = player.getMainHandItem();
        if (!mainHand.is(ModItems.TWILIGHT_SWORD.get())) return;

        if (!player.level().isClientSide && player.level() instanceof ServerLevel serverLevel) {
            Vec3 pos = event.getEntity().position();

            serverLevel.sendParticles(
                    ParticleTypes.DRAGON_BREATH,
                    pos.x, pos.y + 1.0, pos.z,
                    10, // count
                    0.3, 0.3, 0.3,
                    0.01 // speed
            );
        }
    }
}

